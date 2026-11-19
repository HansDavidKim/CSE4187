"""Explicit Hybrid execution with an opt-in isolated highspy backend.

No public entry point accepts prepared state or a caller-constructed QualityProblem.
The only cached bound is local to one validated cohort and one budget invocation.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal
import math
import numbers
import hashlib
import importlib.util
import json
import os
import uuid
from time import perf_counter

import pandas as pd

from . import budget as engine
from . import budget_quality as quality
from .validation import InputValidationError


@dataclass(frozen=True)
class HybridConfig:
    epsilon: object = "0.0001"
    local_improvement: bool = True
    pool_size: int = 32
    max_passes: int = 2
    time_limit: float = 0.5
    backend: str | None = None
    solver_time_limit: float = 3.0
    external_timeout: float = 30.0
    presolve: bool = False
    threads: int = 1
    mip_rel_gap: float = 0.0

    def __post_init__(self):
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (str, Decimal, Fraction, int)):
            raise InputValidationError("hybrid.epsilon: exact string/Decimal/Fraction/integer required")
        try:
            epsilon = Fraction(self.epsilon)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise InputValidationError("hybrid.epsilon: finite rational required") from exc
        if not 0 <= epsilon <= 1:
            raise InputValidationError("hybrid.epsilon: requires 0 <= epsilon <= 1")
        if not isinstance(self.local_improvement, bool):
            raise InputValidationError("hybrid.local_improvement: bool required")
        for field, maximum in (("pool_size", 64), ("max_passes", 8)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= maximum:
                raise InputValidationError(f"hybrid.{field}: integer from 0 through {maximum} required")
        if (isinstance(self.time_limit, bool) or not isinstance(self.time_limit, numbers.Real)
                or not math.isfinite(self.time_limit) or not 0 <= self.time_limit <= 5):
            raise InputValidationError("hybrid.time_limit: finite exchange-search seconds from 0 through 5 required")
        if self.backend not in (None, "highspy"):
            raise InputValidationError("hybrid.backend: None or 'highspy' required")
        for field, maximum in (("solver_time_limit", 3600), ("external_timeout", 3600), ("mip_rel_gap", 1)):
            value = getattr(self, field)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
                    or not 0 <= value <= maximum):
                raise InputValidationError(f"hybrid.{field}: finite value from 0 through {maximum} required")
        if type(self.presolve) is not bool or type(self.threads) is not int or not 1 <= self.threads <= 64:
            raise InputValidationError("hybrid: bool presolve and integer threads from 1 through 64 required")


def _exact(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _run_backend(problem, best, config, certify, binary_tolerance):
    """Only this optional boundary handles worker failures. Input/parent validation stays strict."""
    started = perf_counter()
    evidence = {"requested": True, "backend": "highspy", "status": "skipped", "reason": None,
        "child_created": False, "solver_executed": False, "initial_solution_submitted": False,
        "initial_solution_api_status": None, "initial_solution_acceptance": "unknown",
        "initial_solution_acceptance_evidence": None, "candidate_valid": False, "candidate_adopted": False}
    try:
        if best.meets(config.epsilon):
            evidence["reason"] = "quality_target_met"
        elif not config.external_timeout:
            evidence["reason"] = "additional_time_budget_exhausted"
        elif os.name != "nt":
            evidence["status"] = "unsupported_platform"
        elif importlib.util.find_spec("highspy") is None:
            evidence["status"] = "backend_not_installed"
        else:
            from ._highspy_worker import VERSION, validate_response
            from ._solver_process import run_worker
            indices = problem.active
            ids = [problem.customer_ids[i] for i in indices]
            order_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            selected = set(best.selected_customer_ids)
            request = {"protocol_version": VERSION, "request_id": uuid.uuid4().hex,
                "input_fingerprint": problem.input_sha256, "variable_order_fingerprint": order_hash,
                "n": len(indices), "customer_ids": ids,
                "scores_hex": [float(problem.scores[i]).hex() for i in indices],
                "weights": list(problem.scaled_costs), "capacity": problem.scaled_budget,
                "initial": [int(uid in selected) for uid in ids],
                "options": {"time_limit": float(config.solver_time_limit), "presolve": config.presolve,
                            "threads": config.threads, "mip_rel_gap": float(config.mip_rel_gap), "submit_initial": True}}
            tick = perf_counter()
            transport = run_worker(request, float(config.external_timeout))
            evidence.update(status=transport["status"], child_created=transport["child_created"],
                solver_executed=transport.get("solver_executed"),
                transport={k: v for k, v in transport.items() if k not in ("response", "progress")})
            evidence["worker_external_seconds"] = perf_counter()-tick
            progress = transport.get("progress")
            if progress:
                for key in ("initial_solution_submitted", "initial_solution_api_status",
                            "initial_solution_acceptance", "initial_solution_acceptance_evidence"):
                    evidence[key] = progress[key]
            if transport["status"] == "completed":
                try:
                    response = validate_response(transport["response"], request)
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    evidence.update(status="invalid_response", reason=str(exc))
                else:
                    for key in ("solver_executed", "initial_solution_submitted", "initial_solution_api_status",
                                "initial_solution_acceptance", "initial_solution_acceptance_evidence"):
                        evidence[key] = response[key]
                    evidence["solver"] = {k: v for k, v in response.items() if k not in ("vector",)}
                    evidence["solver"]["objective_semantics"] = "normalized_negative_active_scores; excludes free customers"
                    evidence["solver"]["objective_scale_hex"] = float(max(problem.scores[i] for i in indices)).hex()
                    tick = perf_counter()
                    try:
                        if response["vector"] is None:
                            raise InputValidationError("backend returned no candidate")
                        bits = []
                        for value in response["vector"]:
                            bit = round(value)
                            if bit not in (0, 1) or abs(value-bit) > binary_tolerance:
                                raise InputValidationError("backend returned invalid binary solution")
                            bits.append(bit)
                        selected_ids = [problem.customer_ids[i] for i in problem.free]
                        selected_ids.extend(uid for uid, bit in zip(ids, bits) if bit)
                        # Original Decimal spending and exact binary64 sum, cached exact U.
                        candidate = certify(selected_ids)
                        evidence.update(candidate_valid=True, candidate_exact_score=_exact(candidate.lower_bound),
                                        candidate_expected_cost=str(candidate.total_expected_cost))
                        if candidate.lower_bound > best.lower_bound:
                            best = candidate
                            evidence["candidate_adopted"] = True
                        else:
                            evidence["reason"] = "candidate_not_strictly_better"
                    except (InputValidationError, ValueError, TypeError, OverflowError) as exc:
                        evidence.update(status="invalid_solution", reason=str(exc))
                    evidence["candidate_certification_seconds"] = perf_counter()-tick
    except Exception as exc:
        # Worker availability/transport/runtime failures never replace the held certificate.
        evidence.update(status="launch_error", reason=f"{type(exc).__name__}: {exc}")
    evidence["elapsed_seconds"] = perf_counter()-started
    return best, evidence


def _verify_decisions(prepared, allocation, certify):
    """Bind the actual output back to all prepared rows, costs, and exclusions."""
    rows, summary = allocation.decisions, allocation.summary
    expected = pd.concat([prepared["rows"][list(engine.ROW_COLUMNS)],
                          prepared["excluded"].reindex(columns=engine.ROW_COLUMNS)], ignore_index=True)
    if (rows.customer_id.duplicated().any() or not rows.selected.isin([0, 1]).all()
            or len(rows) != len(expected)):
        raise InputValidationError("hybrid: invalid customer coverage or binary selection")
    try:
        pd.testing.assert_frame_equal(rows[list(engine.ROW_COLUMNS)].reset_index(drop=True), expected,
                                      check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise InputValidationError("hybrid: output context/score differs from validated input") from exc
    n = len(prepared["rows"])
    candidates, excluded = rows.iloc[:n], rows.iloc[n:]
    if (list(candidates.expected_cost) != list(prepared["costs"])
            or candidates.exclusion_reason.notna().any()
            or not excluded.selected.eq(0).all() or not excluded.expected_cost.isna().all()
            or not excluded.selection_reason.eq("upstream_excluded").all()
            or list(excluded.exclusion_reason) != list(prepared["excluded"].exclusion_reason)):
        raise InputValidationError("hybrid: output costs/exclusions differ from validated input")
    selected = candidates.loc[candidates.selected.eq(1)]
    certificate = certify(selected.customer_id.tolist())
    if (summary["has_valid_solution"] is not True or summary["budget_compliant"] is not True
            or summary["budget"] != prepared["budget"]
            or summary["selected_customer_count"] != len(selected)
            or summary["total_expected_cost"] != certificate.total_expected_cost
            or summary["remaining_budget"] != prepared["budget"] - certificate.total_expected_cost
            or summary["objective_value"] != engine._score_sum(selected.priority_score)):
        raise InputValidationError("hybrid: output summary differs from verified selection")
    return certificate


def _run_hybrid(prepared, solver_config, config):
    start = perf_counter()
    timings, attempts = {}, []
    tick = perf_counter()
    baseline = engine._execute(prepared, "greedy", solver_config)
    timings["greedy_seconds"] = perf_counter() - tick
    # Greedy already validates original Decimal spending before it returns.
    timings["first_feasible_seconds"] = perf_counter() - start if baseline.summary["has_valid_solution"] else None
    attempts.append({"stage": "greedy", "status": "completed" if baseline.summary["has_valid_solution"] else "unavailable"})
    epsilon = Fraction(config.epsilon)
    declaration = {"epsilon": _exact(epsilon), "local_improvement": config.local_improvement,
                   "pool_size": int(config.pool_size), "max_passes": int(config.max_passes),
                   "exchange_time_limit": float(config.time_limit), "backend": None,
                   "planning_policy": "no_automatic_approval"}
    if config.backend is not None:
        declaration.update(backend=config.backend, solver_time_limit=float(config.solver_time_limit),
            external_timeout=float(config.external_timeout), presolve=config.presolve,
            threads=config.threads, mip_rel_gap=float(config.mip_rel_gap))
    if not baseline.summary["has_valid_solution"]:
        allocation = deepcopy(baseline)
        allocation.decisions["algorithm"] = "hybrid"
        candidates = allocation.decisions.selection_reason.ne("upstream_excluded")
        allocation.decisions.loc[candidates, "selected"] = pd.NA
        allocation.decisions.loc[candidates, "selection_reason"] = "unavailable"
        allocation.summary.update(algorithm="hybrid", algorithm_version="budget-hybrid-v1", status="unavailable",
            accepted_for_planning=False, optimality="not_proven", objective_value=None,
            selected_customer_count=None, total_expected_cost=None, remaining_budget=None, budget_compliant=None)
        return allocation, {"config": declaration, "quality": {"status": "unavailable", "certificate": None,
            "epsilon": _exact(epsilon), "proven_optimal": False, "epsilon_certified": False},
            "selection": {"initial_greedy_score": None, "final_score": None, "source": None,
                          "improvement": None, "local_improvement_executed": False},
            "attempts": attempts, "timings": timings, "additional_solver_used": False}
    tick = perf_counter()
    problem = quality._from_prepared(prepared)
    timings["quality_preparation_seconds"] = perf_counter() - tick
    tick = perf_counter()
    upper = quality.fractional_upper_bound(problem)
    timings["upper_bound_seconds"] = perf_counter() - tick
    certify = lambda ids: quality._certify_selection(problem, ids, upper)
    tick = perf_counter()
    initial = _verify_decisions(prepared, baseline, certify)
    timings["initial_certification_seconds"] = perf_counter() - tick
    best, source = initial, "greedy"
    attempts.append({"stage": "initial_certification", "status": "target_met" if initial.meets(epsilon) else "target_not_met"})
    run_improvement = config.local_improvement and not initial.meets(epsilon)
    timings["local_improvement_seconds"] = timings["recertification_seconds"] = 0.0
    if run_improvement:
        tick = perf_counter()
        try:
            # Same implementation as public improve_selection, with this invocation's
            # exact upper bound reused at both internal certification points.
            candidate, detail = quality._improve_selection(problem, initial.selected_customer_ids,
                pool_size=config.pool_size, max_passes=config.max_passes, time_limit=config.time_limit,
                certify=certify)
            timings["local_improvement_seconds"] = perf_counter() - tick
            attempts.append({"stage": "local_improvement", "status": "completed", "details": detail})
            tick = perf_counter()
            # Never trust a returned certificate's claimed objective, bound, or cost.
            candidate = certify(candidate.selected_customer_ids)
            timings["recertification_seconds"] = perf_counter() - tick
            if candidate.lower_bound > best.lower_bound:
                best, source = candidate, "local_improvement"
            attempts.append({"stage": "recertification", "status": "improved" if source != "greedy" else "retained_previous"})
        except (InputValidationError, ValueError, TypeError, AttributeError) as exc:
            if not timings["local_improvement_seconds"]:
                timings["local_improvement_seconds"] = perf_counter() - tick
            attempts.append({"stage": "local_improvement_validation", "status": "rejected",
                             "reason": f"{type(exc).__name__}: {exc}"})
    else:
        attempts.append({"stage": "local_improvement", "status": "skipped",
                         "reason": "quality_target_met" if initial.meets(epsilon) else "disabled"})
    backend = None
    if config.backend is not None:
        best, backend = _run_backend(problem, best, config, certify, solver_config.binary_tolerance)
        timings["backend_seconds"] = backend["elapsed_seconds"]
        attempts.append({"stage": "highspy", **backend})
        if backend["candidate_adopted"]:
            source = "highspy"
    tick = perf_counter()
    status = "proven_optimal" if best.optimality_proven else "epsilon_certified" if best.meets(epsilon) else "feasible_uncertified"
    decisions = baseline.decisions.copy(deep=True)
    decisions["algorithm"] = "hybrid"
    decisions["selected"] = pd.array(decisions.customer_id.isin(best.selected_customer_ids).astype(int), dtype="Int64")
    if source != "greedy":
        candidate_mask = decisions.customer_id.isin(problem.customer_ids)
        decisions.loc[candidate_mask & decisions.selected.eq(1), "selection_reason"] = "selected_" + source
        decisions.loc[candidate_mask & decisions.selected.eq(0), "selection_reason"] = "not_selected_hybrid"
    summary = deepcopy(baseline.summary)
    selected_rows = decisions.loc[decisions.selected.eq(1)]
    summary.update(algorithm="hybrid", algorithm_version="budget-hybrid-v1", status=status,
        solver_status=None, solver_success=None, solver_message="No additional solver; exact quality certification",
        accepted_for_planning=False, optimality="exact_optimal" if best.optimality_proven else "not_proven",
        objective_value=engine._score_sum(selected_rows.priority_score), selected_customer_count=len(selected_rows),
        total_expected_cost=best.total_expected_cost, remaining_budget=problem.budget-best.total_expected_cost,
        optimality_gap=None, gap_within_tolerance=None, solver_dual_bound=None, objective_upper_bound=None,
        solver_node_count=None, solver_time=0.0)
    allocation = engine.AllocationResult(decisions, summary)
    if backend is not None:
        summary["solver_message"] = "Optional backend diagnostics in metadata.hybrid.backend; summary quality is exact certification"
    timings["result_construction_seconds"] = perf_counter() - tick
    tick = perf_counter()
    final = _verify_decisions(prepared, allocation, certify)
    if final != best:
        raise InputValidationError("hybrid: final selection differs from retained certificate")
    timings["final_validation_seconds"] = perf_counter() - tick
    attempts.append({"stage": "final_validation", "status": "completed"})
    tick = perf_counter()
    metadata = {"config": declaration,
        "quality": {"status": status, "certificate": final.to_dict(), "epsilon": _exact(epsilon),
                    "proven_optimal": final.optimality_proven, "epsilon_certified": final.meets(epsilon),
                    "method": "exact_binary64_fractional_bound"},
        "selection": {"initial_greedy_score": _exact(initial.lower_bound), "final_score": _exact(final.lower_bound),
                      "initial_greedy_customer_ids": list(initial.selected_customer_ids),
                      "source": source, "selected_count": len(final.selected_customer_ids),
                      "total_expected_cost": str(final.total_expected_cost),
                      "improvement": _exact(final.lower_bound-initial.lower_bound),
                      "local_improvement_executed": run_improvement},
        "attempts": attempts, "timings": timings, "additional_solver_used": False}
    if backend is not None:
        metadata["backend"] = backend
        # None explicitly means execution could not be confirmed (e.g. early crash).
        metadata["additional_solver_used"] = backend["solver_executed"]
    timings["result_construction_seconds"] += perf_counter() - tick
    timings["execution_seconds"] = perf_counter() - start
    summary.update(execution_time=timings["execution_seconds"], memory_after=engine._memory())
    return allocation, metadata
