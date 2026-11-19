"""Post-hoc Greedy certification; never optimizes or changes existing results."""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from time import perf_counter

import pandas as pd

from . import budget as engine
from .budget_quality import prepare_quality_problem, certify_selection
from .validation import CONTEXT_COLUMNS, InputValidationError
from .what_if import WhatIfResult, build_budget_scenarios, SCENARIOS


@dataclass(frozen=True)
class BudgetQualityResult:
    certification_status: str
    certificate: object
    epsilon: Fraction
    metadata: dict
    timings: dict
    reason: str | None = None

    def to_dict(self):
        certificate = self.certificate
        return {"certification_status": self.certification_status, "reason": self.reason,
                "epsilon": {"numerator": str(self.epsilon.numerator), "denominator": str(self.epsilon.denominator)},
                "certified": certificate.meets(self.epsilon) if certificate else False,
                "proven_optimal": certificate.optimality_proven if certificate else False,
                "budget_compliant": True if certificate else None,
                "certificate": certificate.to_dict() if certificate else None,
                "metadata": deepcopy(self.metadata), "timings": dict(self.timings)}


def _fail(message):
    raise InputValidationError("quality_integration: " + message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, default=str)


def _check_rows(actual, expected, columns):
    if not isinstance(actual, pd.DataFrame) or actual.columns.duplicated().any() or not set(columns) <= set(actual):
        _fail("decision columns missing/duplicated")
    a, b = (f[list(columns)].copy().sort_values("customer_id").reset_index(drop=True) for f in (actual, expected))
    for frame in (a, b):
        frame["as_of_at"] = frame.as_of_at.map(lambda v: engine._clock(v, "decision.as_of_at"))
        frame["amount_unit"] = frame.amount_unit.map(lambda v: engine._decimal(v, "decision.amount_unit", positive=True))
        for column in (*engine.PREDICTION_COLUMNS.values(), "priority_score"):
            if column in frame:
                try:
                    frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
                except (ValueError, TypeError):
                    _fail("invalid decision score")
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        _fail("decisions differ from original input: " + str(exc).splitlines()[0])


def certify_budget_result(segmentation, costs, budget, cost_metadata, comparison, *,
                          dataset_id, as_of_at, epsilon="0.0001"):
    """Return separate evidence. Input/result contract errors raise InputValidationError.

    Missing or explicitly invalid Greedy results return unavailable. Selection
    and bound calculation remain together inside the unchanged certify_selection.
    """
    start = perf_counter()
    problem = prepare_quality_problem(segmentation, costs, budget, cost_metadata,
                                      dataset_id=dataset_id, as_of_at=as_of_at)
    prepared_at = perf_counter()
    # Reuse the existing epsilon validator without constructing QualityProblem.
    from .budget_quality import QualityCertificate
    QualityCertificate(Fraction(), Fraction(), Fraction(), problem.budget * 0, (), problem.input_sha256).meets(epsilon)
    epsilon = Fraction(epsilon)
    if not isinstance(comparison, engine.BudgetComparison):
        _fail("BudgetComparison required")
    metadata = json.loads(problem.input_metadata_json)
    for key in ("dataset_id", "as_of_at", "currency", "amount_unit", "cost", "segmentation", "candidate_count", "excluded_count"):
        if _canonical(comparison.metadata.get(key)) != _canonical(metadata[key]):
            _fail("result metadata mismatch: " + key)
    context = {k: metadata[k] for k in ("dataset_id", "as_of_at", "currency", "amount_unit", "cost", "candidate_count", "excluded_count")}
    context.update(algorithm="greedy", budget=str(problem.budget), input_sha256=problem.input_sha256,
                   input_customer_count=len(problem.customer_ids) + metadata["excluded_count"])
    allocation = comparison.results.get("greedy")
    if any(item.summary.get("budget") != problem.budget for item in comparison.results.values()):
        _fail("original result budget mismatch")
    def unavailable(reason):
        return BudgetQualityResult("unavailable", None, epsilon, context,
            {"problem_preparation_seconds": prepared_at-start, "total_seconds": perf_counter()-start}, reason)
    if allocation is None:
        return unavailable("greedy_result_missing")
    summary, rows = allocation.summary, allocation.decisions
    if summary.get("algorithm") != "greedy" or summary.get("budget") != problem.budget:
        _fail("algorithm or original budget mismatch")
    if summary.get("has_valid_solution") is not True or summary.get("status") != "baseline_feasible":
        return unavailable("greedy_result_invalid")
    if not isinstance(rows, pd.DataFrame) or rows.columns.duplicated().any() or not {"customer_id", "selected", "expected_cost", "selection_reason", "algorithm", "exclusion_reason"} <= set(rows):
        _fail("missing decision fields")
    if rows.customer_id.duplicated().any() or not rows.selected.isin([0, 1]).all() or not rows.algorithm.eq("greedy").all():
        _fail("duplicate ID, nonbinary selection, or wrong algorithm")
    point = pd.Timestamp(metadata["as_of_at"])
    choose = lambda f: f.loc[f.dataset_id.eq(dataset_id) & f.as_of_at.map(lambda v: engine._clock(v, "as_of_at")).eq(point)]
    source, excluded = choose(segmentation.customers), choose(segmentation.excluded)
    if set(rows.customer_id) != set(source.customer_id) | set(excluded.customer_id):
        _fail("missing/unknown decision customers")
    candidates = rows.loc[rows.customer_id.isin(problem.customer_ids)]
    omitted = rows.loc[rows.customer_id.isin(excluded.customer_id)]
    _check_rows(candidates, source, engine.ROW_COLUMNS)
    _check_rows(omitted, excluded, (*CONTEXT_COLUMNS, "exclusion_reason"))
    if candidates.selection_reason.eq("upstream_excluded").any() or candidates.exclusion_reason.notna().any():
        _fail("candidate exclusion changed")
    if not omitted.selected.eq(0).all() or not omitted.expected_cost.isna().all() or not omitted.selection_reason.eq("upstream_excluded").all():
        _fail("upstream exclusion changed")
    actual_costs = candidates.set_index("customer_id").expected_cost
    if any(engine._decimal(actual_costs[uid], "decision.expected_cost") != value for uid, value in zip(problem.customer_ids, problem.costs)):
        _fail("decision costs differ from input")
    selected = candidates.loc[candidates.selected.eq(1), "customer_id"].tolist()
    checked_at = perf_counter()
    certificate = certify_selection(problem, selected)
    certified_at = perf_counter()
    selected_scores = candidates.loc[candidates.selected.eq(1), "priority_score"]
    if (summary.get("selected_customer_count") != len(selected)
            or summary.get("total_expected_cost") != certificate.total_expected_cost
            or summary.get("objective_value") != engine._score_sum(selected_scores)
            or summary.get("remaining_budget") != problem.budget-certificate.total_expected_cost
            or summary.get("budget_compliant") is not True):
        _fail("Greedy summary disagrees with verified selection")
    context.update(selected_customer_count=len(selected),
        snapshots=rows[list(CONTEXT_COLUMNS)].drop_duplicates().assign(as_of_at=lambda f: f.as_of_at.map(lambda v: engine._clock(v, "as_of_at").isoformat())).to_dict("records"),
        selection_fingerprint=hashlib.sha256(_canonical([problem.input_sha256, sorted(selected)]).encode()).hexdigest())
    status = "proven_optimal" if certificate.optimality_proven else "epsilon_certified" if certificate.meets(epsilon) else "feasible_uncertified"
    return BudgetQualityResult(status, certificate, epsilon, context,
        {"problem_preparation_seconds": prepared_at-start, "result_binding_seconds": checked_at-prepared_at,
         "selection_and_upper_bound_seconds": certified_at-checked_at,
         "result_construction_seconds": perf_counter()-certified_at, "total_seconds": perf_counter()-start})


def certify_what_if_result(segmentation, costs, base_budget, cost_metadata, result, *,
                           dataset_id, as_of_at, epsilon="0.0001"):
    """Certify only executed scenarios, each with an independently prepared budget."""
    if not isinstance(result, WhatIfResult):
        _fail("WhatIfResult required")
    budgets = build_budget_scenarios(base_budget, cost_metadata)
    if not result.results or list(result.results) != result.metadata.get("executed_scenarios"):
        _fail("scenario result/metadata mismatch")
    for frame in (result.summary, result.decisions):
        if not {"scenario_id", "budget", "budget_multiplier"} <= set(frame) or (len(frame) and set(frame.scenario_id) != set(result.results)):
            _fail("scenario tables disagree with results")
    output = {}
    for name, comparison in result.results.items():
        if name not in budgets:
            _fail("unknown scenario")
        declaration = result.metadata.get("scenario_budgets", {}).get(name)
        if declaration != {"budget_multiplier": dict(SCENARIOS)[name], "budget": budgets[name].amount}:
            _fail("scenario budget mismatch")
        for frame in (result.summary, result.decisions):
            rows = frame.loc[frame.scenario_id.eq(name)]
            if not rows.budget.eq(budgets[name].amount).all() or not rows.budget_multiplier.eq(dict(SCENARIOS)[name]).all():
                _fail("scenario table budget mismatch")
        certified = certify_budget_result(segmentation, costs, budgets[name], cost_metadata, comparison,
            dataset_id=dataset_id, as_of_at=as_of_at, epsilon=epsilon)
        certified.metadata.update(scenario=name, budget_multiplier=str(dict(SCENARIOS)[name]))
        output[name] = certified
    return output
