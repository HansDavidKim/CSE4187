"""Validate a proposed internal five-arm ledger contract using supplied facts.

All five frames use CONTEXT, with the pre-assignment raw SHA on population,
assignments and decisions, and the result raw SHA on executions and outcomes.
Amounts are counts of the declared minor unit. Expected costs and budgets may
be fractional; recorded money must be integral. Nullable fields are explicit.
This contract checks consistency of the supplied records, not RCT authenticity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
import math
from numbers import Integral
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .budget import MAX_DECIMAL_PLACES, _clock, _decimal, _ticks
from .experiment_power import PowerPlan
from .experiment_spec import ExperimentSpec, _number
from .inputs import load_feature_store
from .validation import InputValidationError, _frame, _strings, _times, validate_feature_frame


CONTRACT_VERSION = "kit-five-arm-ledger-v1"
CONTEXT = ("evaluation_dataset_id", "customer_id", "snapshot_id", "as_of_at",
           "raw_database_sha256", "currency", "amount_unit")
LEDGERS = {
    "population": ("eligible", "eligibility_at", "exclusion_reason"),
    "assignments": ("arm_id", "assigned_at", "assignment_probability"),
    "decisions": ("arm_id", "decision_at", "candidate", "selected", "send_requested",
                  "planned_sent_at", "expected_cost"),
    "executions": ("arm_id", "actual_sent", "sent_at", "coupon_id", "expires_at",
                   "discount_rate", "execution_status", "failure_reason", "send_cost_minor"),
    "outcomes": ("arm_id", "outcome_start_at", "outcome_end_at", "outcome_observed",
                 "censoring_reason", "purchase_occurred", "session_occurred",
                 "paid_minor", "coupon_discount_minor", "send_cost_minor"),
}
STRATEGY_FIELDS = ("strategy_version", "model_version", "score_definition_id",
                   "cost_model_version", "solver_kind", "solver_version")


@dataclass(frozen=True)
class ExperimentValidationResult:
    contract_version: str
    experiment_id: str
    experiment_version: str
    evaluation_dataset_id: str
    as_of_at: str
    currency: str
    amount_unit: str
    evaluation_population: int
    eligible_population: int
    by_arm: dict
    readiness_warnings: tuple[str, ...]
    data_integrity: str
    design_status: str
    unresolved_policy_items: tuple[str, ...]
    evidence_status: dict[str, str]

    def to_dict(self) -> dict:
        """Return a detached JSON-ready report with exact decimal amounts as strings."""
        return asdict(self)


def _require(condition, message):
    if not condition:
        raise InputValidationError(f"experiment_run: {message}")


def _text(value, field):
    _require(isinstance(value, str) and bool(value.strip()), f"{field} requires nonempty text")
    return str(value)


def _mapping(value, field):
    _require(isinstance(value, Mapping), f"{field} requires a mapping")
    return value


def _null(value):
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))


def _flag(value, field):
    _require(isinstance(value, (bool, np.bool_)), f"{field} requires a boolean")
    return bool(value)


def _integer(value, field):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
             and value >= 0, f"{field} requires a nonnegative integer")
    return int(value)


def _sha(value, field):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{field} requires a lowercase SHA256 digest")
    return value


def _time(value, field):
    return _clock(value, f"experiment_run.{field}")


def _money(value, field):
    return _ticks(_decimal(value, field), MAX_DECIMAL_PLACES, field)


def _amount_string(ticks):
    # Integer arithmetic keeps the exact aggregate even outside Decimal's default context.
    whole, fraction = divmod(ticks, 10**MAX_DECIMAL_PLACES)
    return f"{whole}.{fraction:0{MAX_DECIMAL_PLACES}d}".rstrip("0").rstrip(".")


def _ledger(frame, name, manifest, point):
    columns = (*CONTEXT, *LEDGERS[name])
    if name != "population" and isinstance(frame, pd.DataFrame) and frame.empty:
        _require(frame.columns.is_unique and set(frame) == set(columns)
                 and frame.attrs.get("evaluation_only") is None,
                 f"{name}: invalid empty ledger columns or attributes")
        return {}
    frame = _frame(frame, columns, columns, name)
    _strings(frame, ("customer_id", "snapshot_id", "currency", "evaluation_dataset_id",
                     "raw_database_sha256"), name)
    _require(not frame.customer_id.duplicated().any(), f"{name}: duplicate customer or reassignment")
    frame["as_of_at"] = _times(frame.as_of_at, f"{name}.as_of_at")
    sha_key = "result_raw_sha256" if name in ("executions", "outcomes") else "feature_raw_sha256"
    for key, expected in (("evaluation_dataset_id", manifest["evaluation_dataset_id"]),
                          ("raw_database_sha256", manifest[sha_key]),
                          ("currency", manifest["currency"]), ("as_of_at", point)):
        _require(frame[key].eq(expected).all(), f"{name}.{key} disagrees with manifest")
    unit = _decimal(manifest["amount_unit"], "amount_unit", positive=True)
    _require(all(_decimal(v, "amount_unit", positive=True) == unit for v in frame.amount_unit),
             f"{name}.amount_unit disagrees with manifest")
    return {(row.customer_id, row.snapshot_id): row for row in frame.itertuples(index=False)}


def _quotas(arms, weights, count):
    """Largest remainders, ties resolved in ExperimentSpec arm order."""
    total = sum(weights)
    divisions = [divmod(count * weight, total) for weight in weights]
    quotas = [q for q, _ in divisions]
    order = sorted(range(len(arms)), key=lambda i: -divisions[i][1])
    for i in order[:count - sum(quotas)]:
        quotas[i] += 1
    return dict(zip(arms, quotas))


def _plan_counts(plan, spec, arms):
    if plan is None:
        return None
    _require(isinstance(plan, PowerPlan), "power_plan requires PowerPlan")
    expected = (spec.experiment_id, spec.version, spec.primary_metric, spec.family_alpha,
                spec.family_alpha / 2, spec.target_power)
    actual = (plan.experiment_id, plan.version, plan.primary_metric, plan.family_alpha,
              plan.comparison_alpha, plan.target_power)
    _require(actual == expected, "power_plan context differs from ExperimentSpec")
    result = []
    for entries in (plan.required_assigned_by_arm, plan.minimum_analyzable_by_arm):
        _require(isinstance(entries, tuple) and len(entries) == len(arms)
                 and all(isinstance(entry, tuple) and len(entry) == 2
                         and isinstance(entry[0], str) for entry in entries),
                 "power_plan requires one count per arm")
        counts = dict(entries)
        _require(set(counts) == set(arms), "power_plan requires one count per arm")
        counts = {arm: _integer(n, "power_plan.count") for arm, n in counts.items()}
        _require(all(counts.values()), "power_plan counts must be positive")
        result.append(counts)
    assigned, analyzed = result
    _require(all(assigned[a] >= analyzed[a] for a in arms), "power_plan counts disagree")
    for counts in (assigned, analyzed):
        blocks = [divmod(counts[arm], weight) for arm, weight in zip(arms, spec.allocation_weights)]
        _require(not any(remainder for _, remainder in blocks)
                 and len({block for block, _ in blocks}) == 1,
                 "power_plan counts do not follow the declared allocation weights")
    return assigned, analyzed


def _decision(row, assigned_at, start, end, key):
    decision_at = _time(row.decision_at, "decision_at")
    _require(assigned_at <= decision_at <= start,
             f"{key}: decision_at must be between the latest assignment and outcome_start_at")
    candidate = _flag(row.candidate, "candidate")
    selected = _flag(row.selected, "selected")
    requested = _flag(row.send_requested, "send_requested")
    _require((not selected or candidate) and (not requested or selected),
             f"{key}: selection/request requires candidacy/selection")
    _require(row.arm_id != "A" or not (selected or requested), f"{key}: A cannot select or request coupons")
    planned = None
    if requested:
        planned = _time(row.planned_sent_at, "planned_sent_at")
        _require(max(decision_at, start) <= planned < end, f"{key}: invalid planned_sent_at")
    else:
        _require(_null(row.planned_sent_at), f"{key}: planned send without request")
    cost = None if _null(row.expected_cost) else _money(row.expected_cost, "expected_cost")
    _require(not selected or cost is not None, f"{key}: selected customer lacks expected_cost")
    return decision_at, selected, requested, planned, cost


def _execution(row, planned, coverage_end, outcome_end, spec, key):
    sent = _flag(row.actual_sent, "actual_sent")
    status = _text(row.execution_status, "execution_status")
    send_cost = _integer(row.send_cost_minor, "execution.send_cost_minor")
    if sent:
        _require(planned is not None, f"{key}: actual send without request")
        _require(status == "sent" and _null(row.failure_reason), f"{key}: contradictory send status")
        coupon_id = _text(row.coupon_id, "coupon_id")
        sent_at = _time(row.sent_at, "sent_at")
        expires = _time(row.expires_at, "expires_at")
        _require(planned <= sent_at < min(coverage_end, outcome_end), f"{key}: invalid sent_at")
        _require(expires - sent_at == timedelta(days=spec.coupon_validity_days)
                 and _decimal(row.discount_rate, "discount_rate") == Decimal(str(spec.coupon_discount)),
                 f"{key}: wrong coupon rate or validity")
        return status, coupon_id, send_cost

    _require(all(_null(getattr(row, field)) for field in ("coupon_id", "sent_at", "expires_at", "discount_rate"))
             and send_cost == 0, f"{key}: unsent customer has a new coupon or send cost")
    if planned is None:
        _require(status == "not_requested" and _null(row.failure_reason),
                 f"{key}: invalid non-request execution status")
    else:
        _require(status in ("failed", "not_sent"), f"{key}: invalid requested execution status")
        _text(row.failure_reason, "failure_reason")
    return status, None, send_cost


def _observation(row, start, end, coverage_end, coupon_id, send_cost, key):
    _require(_time(row.outcome_start_at, "outcome_start_at") == start
             and _time(row.outcome_end_at, "outcome_end_at") == end,
             f"{key}: outcome window must match the declared 28-day window")
    _require(_integer(row.send_cost_minor, "outcome.send_cost_minor") == send_cost,
             f"{key}: observed send cost disagrees with execution")
    observed = _flag(row.outcome_observed, "outcome_observed")
    if not observed:
        _text(row.censoring_reason, "censoring_reason")
        _require(all(_null(getattr(row, field)) for field in
                     ("purchase_occurred", "session_occurred", "paid_minor", "coupon_discount_minor")),
                 f"{key}: censored final outcomes must be NULL")
        return False, None
    _require(end <= coverage_end and _null(row.censoring_reason), f"{key}: invalid completed observation")
    purchase = _flag(row.purchase_occurred, "purchase_occurred")
    _flag(row.session_occurred, "session_occurred")
    paid = _integer(row.paid_minor, "paid_minor")
    discount = _integer(row.coupon_discount_minor, "coupon_discount_minor")
    _require(purchase or (paid == 0 and discount == 0), f"{key}: money recorded without purchase")
    _require(coupon_id is not None or discount == 0, f"{key}: experimental discount without new coupon")
    return True, discount


def _check_features(features, population, manifest, point):
    frame = (load_feature_store(features) if isinstance(features, (str, Path))
             else validate_feature_frame(features))
    expected = {"dataset_id": manifest["evaluation_dataset_id"], "as_of_at": point,
                "raw_database_sha256": manifest["feature_raw_sha256"], "currency": manifest["currency"]}
    for field, value in expected.items():
        _require(frame[field].eq(value).all(), f"features.{field} disagrees with experiment")
    unit = _decimal(manifest["amount_unit"], "amount_unit", positive=True)
    _require(all(Decimal(value) == unit for value in frame.amount_unit), "features.amount_unit mismatch")
    _require(set(zip(frame.customer_id, frame.snapshot_id)) == set(population),
             "features do not cover the evaluation roster by customer and snapshot")


def _key_set(values, field):
    _require(isinstance(values, (tuple, list, set, frozenset)), f"{field} requires customer/snapshot pairs")
    for key in values:
        _require(isinstance(key, tuple) and len(key) == 2, f"{field} requires customer/snapshot pairs")
        for part in key:
            _text(part, field)
    keys = set(values)
    _require(len(keys) == len(values), f"{field}: duplicate keys")
    return keys


def validate_experiment_run(
    spec: ExperimentSpec, manifest: Mapping, *, population: pd.DataFrame,
    assignments: pd.DataFrame, decisions: pd.DataFrame, executions: pd.DataFrame,
    outcomes: pd.DataFrame, power_plan: PowerPlan | None = None,
    features: pd.DataFrame | str | Path | None = None,
    training_customer_ids=None, policy_evidence: Mapping | None = None,
) -> ExperimentValidationResult:
    """Validate one run of the proposed internal contract and report ITT readiness.

    Manifest fields are experiment_id, version, contract_version, evaluation_dataset_id,
    as_of_at, population_frozen_at, observation_start_at, observation_end_at,
    outcome_start_rule (as_of_at or assigned_at), currency, amount_unit,
    eligibility_policy_version, coupon_policy_version, assignment_algorithm,
    assignment_version, assignment_seed, assignment_method (independent or
    fixed_largest_remainder), feature_raw_sha256, result_raw_sha256, ledger_sha256
    (one declaration per LEDGERS entry), budget_per_customer, and arms.
    The as_of_at start rule requires assignment at that same instant so that
    no pre-assignment events fall into the outcome window.
    Decisions must be fixed after all assignments and by observation start.
    Actual delivery can follow a scheduled send within the observation window.
    Each arms entry declares STRATEGY_FIELDS, budget and decision_artifact_id.
    The active arms' budget is budget_per_customer times their ITT headcount.
    Fixed quotas record the marginal probability quota / eligible_population.
    Optional policy_evidence entries supply artifact_id and selected_keys in the
    same customer/snapshot namespace. Training IDs must use the evaluation ID
    namespace. These checks cover supplied evidence, not its source or completeness.
    """
    _require(isinstance(spec, ExperimentSpec), "spec requires ExperimentSpec")
    manifest = _mapping(manifest, "manifest")
    for field, value in (("experiment_id", spec.experiment_id), ("version", spec.version),
                         ("contract_version", CONTRACT_VERSION)):
        _require(_text(manifest.get(field), field) == value, f"manifest.{field} mismatch")
    for field in ("evaluation_dataset_id", "currency", "eligibility_policy_version",
                  "coupon_policy_version", "assignment_algorithm", "assignment_version"):
        _text(manifest.get(field), field)
    _integer(manifest.get("assignment_seed"), "assignment_seed")
    method = _text(manifest.get("assignment_method"), "assignment_method")
    start_rule = _text(manifest.get("outcome_start_rule"), "outcome_start_rule")
    _require(method in ("independent", "fixed_largest_remainder"), "unsupported assignment_method")
    _require(start_rule in ("as_of_at", "assigned_at"), "unsupported outcome_start_rule")
    point, frozen, coverage_start, coverage_end = (
        _time(manifest.get(field), field) for field in
        ("as_of_at", "population_frozen_at", "observation_start_at", "observation_end_at"))
    _require(coverage_start <= point <= frozen < coverage_end, "invalid observation or cohort freeze clocks")
    unit = _decimal(manifest.get("amount_unit"), "amount_unit", positive=True)
    for field in ("feature_raw_sha256", "result_raw_sha256"):
        _sha(manifest.get(field), field)
    digests = _mapping(manifest.get("ledger_sha256"), "ledger_sha256")
    _require(set(digests) == set(LEDGERS), "ledger_sha256 requires all five ledger declarations")
    for name, digest in digests.items():
        _sha(digest, name)

    arms = tuple(arm for arm, _ in spec.arms)
    strategies = _mapping(manifest.get("arms"), "arms")
    _require(set(strategies) == set(arms), "manifest requires exactly the five strategy arms")
    budgets = {}
    for arm in arms:
        strategy = _mapping(strategies[arm], arm)
        for field in (*STRATEGY_FIELDS, "decision_artifact_id"):
            _text(strategy.get(field), f"{arm}.{field}")
        budgets[arm] = _money(strategy.get("budget"), f"{arm}.budget")
    for field in ("strategy_version", "model_version", "score_definition_id", "cost_model_version"):
        _require(strategies["D1"][field] == strategies["D2"][field], f"D1/D2 {field} must match")
    _require(strategies["D1"]["solver_kind"] == "greedy" and strategies["D2"]["solver_kind"] == "milp",
             "D1/D2 require Greedy/MILP declarations")
    per_customer = _money(manifest.get("budget_per_customer"), "budget_per_customer")
    plan = _plan_counts(power_plan, spec, arms)

    roster = _ledger(population, "population", manifest, point)
    eligible = set()
    for key, row in roster.items():
        eligibility_at = _time(row.eligibility_at, "eligibility_at")
        _require(coverage_start <= eligibility_at <= point, f"{key}: eligibility uses an invalid cutoff")
        if _flag(row.eligible, "eligible"):
            _require(_null(row.exclusion_reason), f"{key}: eligible customer has exclusion reason")
            eligible.add(key)
        else:
            _text(row.exclusion_reason, "exclusion_reason")

    assigned = _ledger(assignments, "assignments", manifest, point)
    decision = _ledger(decisions, "decisions", manifest, point)
    execution = _ledger(executions, "executions", manifest, point)
    observed = _ledger(outcomes, "outcomes", manifest, point)
    for name, ledger in (("assignments", assigned), ("decisions", decision),
                         ("executions", execution), ("outcomes", observed)):
        _require(set(ledger) == eligible, f"{name} keys differ from the common eligible roster")

    quotas = _quotas(arms, spec.allocation_weights, len(eligible))
    target = dict(zip(arms, (Fraction(w, sum(spec.allocation_weights)) for w in spec.allocation_weights)))
    counts = ("itt", "selected", "requested", "sent", "failed", "not_sent", "not_requested",
              "observed", "censored", "expected_ticks", "send_cost_minor", "observed_discount_minor")
    summaries = {arm: dict.fromkeys(counts, 0) for arm in arms}
    selected_keys = {arm: set() for arm in arms}
    assignment_clocks = {key: _time(row.assigned_at, "assigned_at") for key, row in assigned.items()}
    latest_assignment = max(assignment_clocks.values(), default=frozen)
    coupon_ids = set()
    for key, row in assigned.items():
        arm = _text(row.arm_id, "assignments.arm_id")
        _require(arm in summaries, f"{key}: invalid arm_id")
        _require(all(isinstance(r.arm_id, str) and r.arm_id == arm for r in
                     (decision[key], execution[key], observed[key])), f"{key}: arm changed after assignment")
        assigned_at = assignment_clocks[key]
        _require(frozen <= assigned_at < coverage_end, f"{key}: assigned_at outside allowed range")
        if start_rule == "as_of_at":
            _require(assigned_at == point, f"{key}: as_of_at outcome start requires simultaneous assignment")
        expected_p = target[arm] if method == "independent" else Fraction(quotas[arm], len(eligible))
        probability = _number(row.assignment_probability, "assignment_probability")
        _require(expected_p > 0 and 0 < probability <= 1
                 and math.isclose(probability, float(expected_p), rel_tol=6e-8),
                 f"{key}: assignment_probability disagrees with the declared design")
        start = point if start_rule == "as_of_at" else assigned_at
        end = start + timedelta(days=spec.outcome_days)
        decision_at, selected, requested, planned, expected_cost = _decision(
            decision[key], latest_assignment, start, end, key)
        _require(decision_at < coverage_end, f"{key}: decision_at exceeds recorded coverage")
        status, coupon_id, send_cost = _execution(execution[key], planned, coverage_end, end, spec, key)
        if coupon_id is not None:
            _require(coupon_id not in coupon_ids, f"{key}: duplicate experimental coupon_id")
            coupon_ids.add(coupon_id)
        complete, discount = _observation(observed[key], start, end, coverage_end, coupon_id, send_cost, key)

        summary = summaries[arm]
        summary["itt"] += 1
        summary["selected"] += int(selected)
        summary["requested"] += int(requested)
        summary[status] += 1
        summary["observed" if complete else "censored"] += 1
        summary["send_cost_minor"] += send_cost
        if discount is not None:
            summary["observed_discount_minor"] += discount
        if selected:
            summary["expected_ticks"] += expected_cost
            selected_keys[arm].add(key)

    warnings = []
    for arm, summary in summaries.items():
        if method == "fixed_largest_remainder":
            _require(summary["itt"] == quotas[arm], f"{arm}: fixed quota count mismatch")
        budget = budgets[arm]
        _require(budget == (0 if arm == "A" else per_customer * summary["itt"]),
                 f"{arm}: budget disagrees with the common per-customer limit")
        expected = summary.pop("expected_ticks")
        _require(expected <= budget, f"{arm}: selected expected costs exceed budget")
        known_spend = summary["send_cost_minor"] + summary["observed_discount_minor"]
        excess = known_spend * 10**MAX_DECIMAL_PLACES > budget
        summary.update(budget=_amount_string(budget), selected_expected_cost=_amount_string(expected),
                       plan_within_budget=True, unsent=summary["itt"] - summary["sent"],
                       known_spend_minor=known_spend,
                       realized_spend_minor=None if summary["censored"] else known_spend,
                       realized_budget_status="over_budget" if excess else
                       "unverified_incomplete" if summary["censored"] else "within_budget",
                       sample_plan=None)
        if plan is not None:
            assigned_target, analyzed_target = (counts[arm] for counts in plan)
            summary["sample_plan"] = {
                "assigned_target": assigned_target, "observed_target": analyzed_target,
                "assigned_shortfall": max(0, assigned_target - summary["itt"]),
                "observed_shortfall": max(0, analyzed_target - summary["observed"]),
            }
            if summary["itt"] < assigned_target or summary["observed"] < analyzed_target:
                warnings.append(f"{arm}: assigned or observed sample is below the PowerPlan")

    evidence = dict.fromkeys(("randomization", "independent_evaluation", "training_overlap",
                             "strategy_approval", "policy_decisions", "actual_policy_application",
                             "causal_outcome_generation", "producer_file_integrity", "feature_context"),
                            "unverified")
    if features is not None:
        _check_features(features, roster, manifest, point)
        evidence["feature_context"] = "supplied_feature_context_matches"
    if training_customer_ids is not None:
        _require(isinstance(training_customer_ids, (tuple, list, set, frozenset)),
                 "training_customer_ids requires IDs in the evaluation namespace")
        training = {_text(uid, "training_customer_ids") for uid in training_customer_ids}
        _require(not training.intersection(uid for uid, _ in roster), "training/evaluation customer overlap")
        evidence["training_overlap"] = "no_overlap_in_supplied_ids"
    if policy_evidence is not None:
        supplied = _mapping(policy_evidence, "policy_evidence")
        _require(bool(supplied) and set(supplied) <= set(arms) - {"A"}, "invalid policy_evidence arms")
        for arm, artifact in supplied.items():
            artifact = _mapping(artifact, "policy_evidence entry")
            _require(_text(artifact.get("artifact_id"), "artifact_id") == strategies[arm]["decision_artifact_id"]
                     and _key_set(artifact.get("selected_keys"), "selected_keys") == selected_keys[arm],
                     f"{arm}: policy evidence differs from the declared decision artifact or selections")
        evidence["policy_decisions"] = ("all_active_arm_selections_match_supplied_evidence"
                                        if set(supplied) == set(arms) - {"A"}
                                        else "partial_arm_selections_match_supplied_evidence")

    return ExperimentValidationResult(
        CONTRACT_VERSION, spec.experiment_id, spec.version, manifest["evaluation_dataset_id"],
        point.isoformat(), manifest["currency"], str(unit), len(roster), len(eligible), summaries,
        tuple(warnings), "passed", spec.design_status, spec.unresolved_policy_items, evidence,
    )
