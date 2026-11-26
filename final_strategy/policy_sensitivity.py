"""One declared perturbation at a time on a fixed human-strategy evaluation cohort."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Context, DecimalException, Inexact, localcontext
from fractions import Fraction
from numbers import Integral
from time import perf_counter

import numpy as np

from . import budget
from .coupon_cost_estimation import CostEstimate
from .policy_comparison import _json, compare_policies
from .strategy_profiles import build_strategy_profiles
from .validation import CONTEXT_COLUMNS, InputValidationError
from .what_if import SCENARIOS, build_budget_scenarios


_COST = ("send_cost", "redemption_probability", "expected_discount_if_redeemed")
_CUTS = ("common", "risk_high", "risk_low", "high_value", "standard_value", "h4_r")


@dataclass(frozen=True)
class SensitivityScenario:
    scenario_id: str
    field: str
    value: object
    reason: str


class SensitivityResult:
    def __init__(self, report):
        self._report = report

    def to_dict(self):
        return deepcopy(_json(self._report))


def _require(condition, message):
    if not condition:
        raise InputValidationError("policy sensitivity: " + message)


def _cost_scenario(original, field, factor, places):
    columns = (_COST[0], _COST[2]) if field == "monetary_components" else (field,)
    table = original.copy()
    try:
        with localcontext(Context(prec=200)) as ctx:
            ctx.traps[Inexact] = True
            for col in columns:
                table[col] = [budget._decimal(budget._decimal(x, col) * factor, col)
                              for x in original[col].astype(object)]
            for send, prob, discount in table[list(_COST)].itertuples(index=False, name=None):
                p = budget._decimal(prob, _COST[1])
                _require(p <= 1, "redemption_probability exceeds one")
                total = budget._decimal(send, _COST[0]) + p * budget._decimal(discount, _COST[2])
                budget._ticks(budget._decimal(total, "expected_cost"), places, "expected_cost")
    except DecimalException as exc:
        raise InputValidationError("inexact or unsupported scenario cost") from exc
    return table


def _difference(before, after):
    difference = None if before is None or after is None else Fraction(after) - Fraction(before)
    return dict(before=before, after=after, change=difference,
                relative_change=None if difference is None or not before else difference / Fraction(before))


def _cut_value(field, value):
    if field == "h4_r":
        return {k: budget._decimal(v, k) for k, v in value.items()}
    return budget._decimal(value, field)


def _sets(before, after):
    if before is None or after is None:
        return dict(intersection=None, added=None, removed=None, jaccard=None)
    union = before | after
    return dict(intersection=len(before & after), added=len(after - before), removed=len(before - after),
                jaccard=Fraction(len(before & after), len(union)) if union else Fraction(1))


def _tables(result):
    """Index representative tables once, including when the owner of an alias changes."""
    plans = result.table("plans").set_index("policy_id").to_dict("index")
    for plan in plans.values():
        if plan["has_valid_solution"]:
            remaining = Fraction(plan["remaining_budget"])
            _require(remaining >= 0 and Fraction(plan["total_expected_cost"]) + remaining == Fraction(plan["budget"]),
                     "valid plan does not reconcile to its budget")
    segments = result.table("segments").groupby("representative_id")
    decisions = result.table("decisions").groupby("representative_id")
    ips = result.table("ips").groupby("policy_id")
    pareto = result.table("pareto").set_index("policy_id").status
    reps = {}
    for rep, frame in decisions:
        keyed = frame.set_index("customer_id").sort_index()
        context = keyed[[c for c in CONTEXT_COLUMNS if c != "customer_id"] + ["segment_id"]]
        reps[rep] = (context, set(keyed.index[keyed.budget_candidate.eq(True)]),
                     set(keyed.index[keyed.selected.eq(1)]),
                     segments.get_group(rep).set_index("segment_id").to_dict("index"))
    return {name: (plan, *reps[plan["representative_id"]], ips.get_group(name).set_index("metric").to_dict("index"),
                   pareto[name]) for name, plan in plans.items()}


def _contrast(base, scenario):
    _require(base.keys() == scenario.keys(), "policy declarations differ")
    rows = {}
    for name, (left, cohort, candidates, selected, groups, ips, pareto) in base.items():
        right, other_cohort, other_candidates, other_selected, other_groups, other_ips, other_pareto = scenario[name]
        _require(cohort.equals(other_cohort) and left["policy_version"] == right["policy_version"],
                 "customer context, classification or policy version differs")
        metrics = {}
        for metric, a in ips.items():
            b = other_ips[metric]
            metrics[metric] = dict(**_difference(a["estimate"], b["estimate"]),
                                   estimable=[a["estimable"], b["estimable"]],
                                   hold_reason=[a["hold_reason"], b["hold_reason"]],
                                   ess=[a.get("effective_sample_size"), b.get("effective_sample_size")])
        rows[name] = dict(
            candidates=_sets(candidates, other_candidates),
            selected=_sets(selected if left["has_valid_solution"] else None,
                           other_selected if right["has_valid_solution"] else None),
            plan={k: _difference(left[k], right[k]) for k in ("candidate_count", "selected_customer_count",
                  "total_expected_cost", "budget_utilization", "objective_value")},
            has_valid_solution=[left["has_valid_solution"], right["has_valid_solution"]],
            accepted_for_planning=[left["accepted_for_planning"], right["accepted_for_planning"]],
            segment_selection_rates={s: _difference(g["population_selection_rate"],
                                      other_groups[s]["population_selection_rate"]) for s, g in groups.items()},
            ips=metrics, pareto_status=[pareto, other_pareto])
    return rows


@contextmanager
def _planning_context():
    # A valid allocation and its remainder are bounded by MAX_AMOUNT. Expressed
    # in the smallest budget tick they need at most 19 + 9 significant digits.
    digits = budget.MAX_AMOUNT.adjusted() + 1 + budget.MAX_DECIMAL_PLACES
    with localcontext(Context(prec=digits)):
        yield


@_planning_context()
def run_policy_sensitivity(*, search_inputs, profile_settings, cost_estimate, scenarios, max_scenarios,
                           pareto_directions, baseline_id="baseline", experiment_log=None, evidence=None,
                           baseline_fixations=None, allow_unapproved=False, allow_unverified_log=False,
                           include_exploratory=False):
    """Compare named scenarios with one baseline. The scenario cap includes the baseline."""
    start = perf_counter()
    budget._text(baseline_id, "baseline_id")
    _require(isinstance(max_scenarios, Integral) and not isinstance(max_scenarios, (bool, np.bool_))
             and max_scenarios >= 1, "positive integer scenario cap required")
    _require(isinstance(scenarios, (tuple, list)) and len(scenarios) < max_scenarios, "scenario cap exceeded")
    changes = deepcopy(scenarios)
    settings = deepcopy(profile_settings)
    profiles = build_strategy_profiles(**settings)
    profile_variants = {}
    cost_factors = {}
    ids = {baseline_id}
    for item in changes:
        _require(type(item) is SensitivityScenario, "SensitivityScenario required")
        budget._text(item.scenario_id, "scenario_id")
        budget._text(item.reason, "reason")
        _require(item.scenario_id not in ids, "duplicate scenario ID")
        budget._text(item.field, "scenario field")
        if item.field == "budget":
            _require(isinstance(item.value, str) and item.value in dict(SCENARIOS), "unknown What-if budget ID")
        elif item.field == "algorithm":
            _require(isinstance(item.value, str) and item.value in ("greedy", "milp", "hybrid"), "unknown algorithm")
        elif item.field in _COST or item.field == "monetary_components":
            cost_factors[item.scenario_id] = budget._decimal(item.value, "cost_multiplier")
        elif item.field in _CUTS:
            profile_variants[item.scenario_id] = build_strategy_profiles(**(settings | {item.field: item.value}))
        else:
            raise InputValidationError("policy sensitivity: unknown change")
        ids.add(item.scenario_id)
    _require(isinstance(cost_estimate, CostEstimate), "CostEstimate required")
    _require(not {"cost_inputs", "cost_metadata"} & search_inputs.keys(), "supply costs only through CostEstimate")
    report = dict(baseline_id=baseline_id, baseline_cost_audit=cost_estimate.audit, scenarios={}, differences={},
                  comparison_calls=0, dispatch_authorized=False, actual_data_verified=False)
    if cost_estimate.status == "withheld":
        return SensitivityResult(dict(report, status="withheld", reason="cost_estimate_withheld",
                                      elapsed_seconds=perf_counter()-start))
    shared = deepcopy(search_inputs)
    shared.update(cost_inputs=cost_estimate.cost_inputs, cost_metadata=cost_estimate.cost_metadata)
    report["original_cost_inputs"] = shared["cost_inputs"].to_dict("records")
    options = deepcopy(dict(pareto_directions=pareto_directions, experiment_log=experiment_log, evidence=evidence,
                            allow_unapproved=allow_unapproved, allow_unverified_log=allow_unverified_log,
                            include_exploratory=include_exploratory))
    budgets = (build_budget_scenarios(shared["budget"], shared["cost_metadata"])
               if any(s.field == "budget" for s in changes) else {})
    baseline_algorithm = shared.get("algorithm", "greedy")
    baseline_tables = None
    for item in [None, *changes]:
        tick = perf_counter()
        inputs = shared.copy()
        candidates = profiles
        identifier = baseline_id if item is None else item.scenario_id
        field = None if item is None else item.field
        before = after = None
        reuse = False
        if field == "budget":
            inputs["budget"] = budgets[item.value]
            before, after = budget._decimal(shared["budget"].amount, "budget"), inputs["budget"].amount
            reuse = item.value == "budget_100"
        elif field in _COST or field == "monetary_components":
            after = cost_factors[identifier]
            inputs["cost_inputs"] = _cost_scenario(shared["cost_inputs"], field, after,
                                                   shared["cost_metadata"].decimal_places)
            before = 1
        elif field in _CUTS:
            before, after = settings[field], item.value
            candidates = profile_variants[identifier]
            before, after = _cut_value(field, before), _cut_value(field, after)
        elif field == "algorithm":
            before, after = baseline_algorithm, item.value
            inputs["algorithm"] = after
            if after != "hybrid":
                inputs.pop("hybrid_config", None)
        if reuse:
            # A reference to the identical baseline, including its original evidence grade.
            record = report["scenarios"][baseline_id]["comparison"]
            tables = baseline_tables
        else:
            result = compare_policies(candidates, search_inputs=inputs, scenario_id=identifier, pairs=[],
                                      fixations=baseline_fixations if item is None else None, **options)
            report["comparison_calls"] += 1
            record = result.to_dict()
            tables = _tables(result)
        if item is None:
            baseline_tables = tables
        else:
            report["differences"][identifier] = _contrast(baseline_tables, tables)
        hybrid = budget._hybrid_config(inputs.get("algorithm", "greedy"), inputs.get("hybrid_config"))
        report["scenarios"][identifier] = dict(
            scenario_id=identifier, baseline_id=baseline_id, field=field, before=before, after=after,
            reason=item.reason if item else None, reused_from=baseline_id if reuse else None,
            hybrid_config=asdict(hybrid) if hybrid else None, comparison=record,
            elapsed_seconds=perf_counter()-tick)
    return SensitivityResult(dict(report, status="complete", elapsed_seconds=perf_counter()-start))
