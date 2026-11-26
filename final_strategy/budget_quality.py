"""Exact, opt-in quality certificates for the existing single-cohort budget model.

This does not change optimize_budget, solver status, or planning approval. Scores
mean the actual binary64 values accepted by budget._prepare, not unknown original
decimal/model values. Fraction fields, never display floats, certify inequalities.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import reduce
from itertools import combinations
import math
import numbers
import hashlib
import json
from time import perf_counter

from . import budget as engine
from .validation import InputValidationError


@dataclass(frozen=True)
class QualityProblem:
    """Immutable numeric view of one validated cohort; all indices are ID-sorted."""

    customer_ids: tuple[str, ...]
    scores: tuple[Fraction, ...]
    costs: tuple[Decimal, ...]
    ticks: tuple[int, ...]
    budget: Decimal
    budget_ticks: int
    active: tuple[int, ...]
    free: tuple[int, ...]
    scaled_costs: tuple[int, ...]
    scaled_budget: int
    input_metadata_json: str
    input_sha256: str


@dataclass(frozen=True)
class QualityCertificate:
    lower_bound: Fraction
    upper_bound: Fraction
    relative_error_bound: Fraction
    total_expected_cost: Decimal
    selected_customer_ids: tuple[str, ...]
    input_sha256: str

    @property
    def optimality_proven(self):
        return self.lower_bound == self.upper_bound

    def meets(self, epsilon):
        # String/Decimal/Fraction are intentional: no hidden epsilon rounding.
        if isinstance(epsilon, bool) or not isinstance(epsilon, (str, Decimal, Fraction, int)):
            raise InputValidationError("quality.epsilon: exact decimal string or rational required")
        try:
            value = Fraction(epsilon)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise InputValidationError("quality.epsilon: finite rational required") from exc
        if not 0 <= value <= 1:
            raise InputValidationError("quality.epsilon: requires 0 <= epsilon <= 1")
        return self.relative_error_bound <= value

    def to_dict(self):
        def exact(value):
            return {"numerator": str(value.numerator), "denominator": str(value.denominator)}

        return {"score_semantics": "exact_binary64_priority_score", "certificate_version": "fractional-v1",
                "input_sha256": self.input_sha256,
                "lower_bound": exact(self.lower_bound), "upper_bound": exact(self.upper_bound),
                "relative_error_bound": exact(self.relative_error_bound),
                "display_objective": float(self.lower_bound),
                "display_upper_bound": float(self.upper_bound),
                "display_relative_error_bound": float(self.relative_error_bound),
                "optimality_proven": self.optimality_proven,
                "epsilon_certified": {e: self.meets(e) for e in ("0.01", "0.001", "0.0001")},
                "total_expected_cost": str(self.total_expected_cost),
                "selected_customer_ids": list(self.selected_customer_ids)}


def _from_prepared(prepared):
    metadata = json.dumps(prepared["metadata"], sort_keys=True, default=str)
    # Include every preserved row identifier, including snapshot/week/raw source,
    # plus model/cost/segmentation declarations. Bind certificates to this input.
    payload = json.dumps({"metadata": metadata, "rows": prepared["rows"].to_dict(orient="records"),
                          "excluded": prepared["excluded"].to_dict(orient="records"),
                          "scores": [float(s).hex() for s in prepared["scores"]],
                          "budget": str(prepared["budget"])}, sort_keys=True, default=str)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return QualityProblem(tuple(prepared["customer_ids"]),
                          tuple(Fraction(*float(s).as_integer_ratio()) for s in prepared["scores"]),
                          tuple(prepared["costs"]), tuple(prepared["ticks"]), prepared["budget"],
                          prepared["budget_ticks"], tuple(prepared["active"]), tuple(prepared["free"]),
                          tuple(prepared["scaled_costs"]), prepared["scaled_budget"], metadata, fingerprint)


def prepare_quality_problem(segmentation, costs, budget, cost_metadata, *, dataset_id, as_of_at):
    """Reuse all source/time/unit/Decimal checks; adds no external model contract."""
    return _from_prepared(engine._prepare(segmentation, costs, cost_metadata, budget, dataset_id, as_of_at))


def _density_order(problem):
    return sorted(problem.active, key=lambda i: (-problem.scores[i] / problem.ticks[i],
                                                -problem.scores[i], problem.customer_ids[i]))


def fractional_upper_bound(problem):
    """Exact LP bound on the equivalent cost-GCD/floored-RHS binary problem.

    Positive free items are included. Nonpositive scores and individually
    over-budget items cannot improve an optimum and are omitted. Relaxing the
    remaining binary variables to [0,1] gives an upper bound, not an integer plan.
    """
    bound = sum((problem.scores[i] for i in problem.free), Fraction())
    remaining = problem.scaled_budget
    weights = dict(zip(problem.active, problem.scaled_costs))
    for i in _density_order(problem):
        weight = weights[i]
        take = min(weight, remaining)
        bound += problem.scores[i] * Fraction(take, weight)
        remaining -= take
        if not remaining:
            break
    return bound


def certify_selection(problem, selected_customer_ids):
    """Certify only a unique, known, policy-admissible, exact-budget-feasible plan.

    Call with a QualityProblem returned by prepare_quality_problem. A certificate
    is not a send authorization and does not approve an unproven MILP result.
    """
    return _certify_selection(problem, selected_customer_ids, fractional_upper_bound(problem))


def _certify_selection(problem, selected_customer_ids, upper):
    """Internal only: upper must be computed for this same call-owned problem.

    Public callers cannot supply a cached bound. Hybrid constructs its problem
    from budget._prepare inside the invocation, never from caller QualityProblem.
    """
    selected = tuple(selected_customer_ids)
    if any(not isinstance(v, str) for v in selected) or len(set(selected)) != len(selected):
        raise InputValidationError("quality.selection: requires unique string customer IDs")
    lookup = {customer: i for i, customer in enumerate(problem.customer_ids)}
    if not set(selected) <= lookup.keys():
        raise InputValidationError("quality.selection: unknown/excluded customer")
    indices = [lookup[customer] for customer in selected]
    if not set(indices) <= set(problem.active) | set(problem.free):
        raise InputValidationError("quality.selection: nonpositive or individually over-budget customer")
    total = engine._decimal_sum(problem.costs[i] for i in indices)
    if total > problem.budget or sum(problem.ticks[i] for i in indices) > problem.budget_ticks:
        raise InputValidationError("quality.selection: exact Decimal budget exceeded")
    lower = sum((problem.scores[i] for i in indices), Fraction())
    if lower > upper:
        raise InputValidationError("quality: lower bound exceeds upper bound")
    error = (upper - lower) / upper if upper else Fraction()
    return QualityCertificate(lower, upper, error, total, tuple(sorted(selected)), problem.input_sha256)


def exact_integer_objective(problem, max_integer=2**63 - 1):
    """Exact active-score conversion for integer-only solvers, or explicit refusal.

    score[i] = coefficient[i] * returned_factor. No rounding. Conservatively
    bound aggregate objective and weight activity for int64 solver arithmetic.
    """
    if (isinstance(max_integer, bool) or not isinstance(max_integer, numbers.Integral)
            or not 1 <= max_integer <= 2**63 - 1):
        raise InputValidationError("quality.max_integer: positive int64 bound required")
    values = [problem.scores[i] for i in problem.active]
    denominator = max((value.denominator for value in values), default=1)
    numerators = [value.numerator * (denominator // value.denominator) for value in values]
    divisor = reduce(math.gcd, numerators, 0) or 1
    coefficients = tuple(value // divisor for value in numerators)
    if (sum(coefficients) > max_integer or sum(problem.scaled_costs) > max_integer
            or problem.scaled_budget > max_integer):
        raise InputValidationError("quality.integer_objective: exact conversion exceeds safe integer range")
    return coefficients, Fraction(divisor, denominator)


def improve_selection(problem, baseline_customer_ids, *, pool_size=32, max_passes=2, time_limit=0.5):
    """Separate bounded heuristic; never modifies or relabels the Greedy baseline.

    Best single + free items, refill, and at most max_passes exchanges of one
    selected item for one/two unselected items within deterministic small pools.
    O(n log n + passes * pool_size**3); time_limit bounds exchange search only,
    not preparation/sorting/certification or the overall API. A time-limited
    run can stop at different improvements; every returned plan is certified.
    """
    return _improve_selection(problem, baseline_customer_ids, pool_size=pool_size,
                              max_passes=max_passes, time_limit=time_limit,
                              certify=lambda ids: certify_selection(problem, ids))


def _improve_selection(problem, baseline_customer_ids, *, pool_size, max_passes, time_limit, certify):
    """Shared improvement implementation; internal Hybrid supplies same-problem certification."""
    started = perf_counter()
    for name, value in (("pool_size", pool_size), ("max_passes", max_passes)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise InputValidationError(f"quality.{name}: nonnegative integer required")
    if isinstance(time_limit, bool) or not isinstance(time_limit, numbers.Real) or not math.isfinite(time_limit) or time_limit < 0:
        raise InputValidationError("quality.time_limit: finite nonnegative value required")
    baseline = certify(baseline_customer_ids)
    lookup = {customer: i for i, customer in enumerate(problem.customer_ids)}
    selected = {lookup[customer] for customer in baseline.selected_customer_ids} | set(problem.free)
    score = lambda choices: sum((problem.scores[i] for i in choices), Fraction())
    if problem.active:
        best = min(problem.active, key=lambda i: (-problem.scores[i], problem.customer_ids[i]))
        single = set(problem.free) | {best}
        if score(single) > score(selected):
            selected = single
    order = _density_order(problem)
    def fill():
        remaining = problem.budget_ticks - sum(problem.ticks[i] for i in selected)
        for i in order:
            if i not in selected and problem.ticks[i] <= remaining:
                selected.add(i)
                remaining -= problem.ticks[i]
        return remaining
    remaining = fill()
    deadline, examined = perf_counter() + time_limit, 0
    iterations, accepted, termination = 0, 0, "iteration_limit"
    for _ in range(max_passes):
        iterations += 1
        incoming = sorted((i for i in problem.active if i not in selected),
                          key=lambda i: (-problem.scores[i], problem.customer_ids[i]))[:pool_size]
        outgoing = sorted((i for i in selected if problem.ticks[i]),
                          key=lambda i: (problem.scores[i], problem.customer_ids[i]))[:pool_size]
        best_change, best_gain = None, Fraction()
        for old in outgoing:
            for count in (1, 2):
                for new in combinations(incoming, count):
                    if perf_counter() >= deadline:
                        break
                    examined += 1
                    gain = score(new) - problem.scores[old]
                    if gain > best_gain and sum(problem.ticks[i] for i in new) <= remaining + problem.ticks[old]:
                        best_change, best_gain = (old, new), gain
                if perf_counter() >= deadline:
                    break
            if perf_counter() >= deadline:
                break
        if best_change is None:
            termination = ("pool_exhausted" if not incoming or not outgoing else
                           "time_limit" if perf_counter() >= deadline else "no_improvement")
            break
        old, new = best_change
        selected.remove(old)
        selected.update(new)
        accepted += 1
        remaining = fill()
        if perf_counter() >= deadline:
            termination = "time_limit"
            break
    certificate = certify([problem.customer_ids[i] for i in selected])
    return certificate, {"algorithm": "bounded-local-v1", "examined_exchanges": examined,
                         "accepted_exchanges": accepted, "iterations": iterations,
                         "termination_reason": termination, "elapsed_seconds": perf_counter()-started,
                         "pool_size": pool_size, "max_passes": max_passes,
                         "exchange_time_limit": time_limit, "baseline_lower_bound": str(baseline.lower_bound)}
