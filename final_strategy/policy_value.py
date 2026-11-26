"""Point estimates for fixed send/no-send plans on one balanced assignment log."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
from numbers import Integral
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import budget
from .policy_search import PolicyCandidate, PolicyRun, _fingerprint
from .retention_strategy import SegmentPolicy
from .segmentation import SegmentationConfig
from .validation import InputValidationError, JOIN_KEYS


ASSIGNMENT_DESIGN = "balanced_fixed_randomized"
INTERVENTION_SPEC = (
    "one 10% coupon at t0 on the highest-unit-price purchased unit, "
    "valid 28d, no later sending"
)
IDENTITY = ("dataset_id", "as_of_at", "customer_id", "snapshot_id", "raw_database_sha256")
MONEY_KEYS = ("currency", "amount_unit")
LOG_COLUMNS = (
    *JOIN_KEYS, "snapshot_id", *MONEY_KEYS, "experiment_id", "assigned_at", "treatment",
    "assignment_probability", "coupon_action", "actual_sent", "coupon_id", "execution_status",
    "sent_at", "expires_at", "discount_rate", "additional_send_count", "eligible_at_assignment",
    "outcome_end_at", "outcome_observed", "factual_no_order", "factual_no_session",
    "factual_outcome_net_revenue_minor", "send_cost_minor", "outcome_redemption_discount_cost_minor",
)
METRICS = (
    "purchase_rate_28d", "no_session_rate_28d", "paid_minor_per_customer",
    "send_cost_minor_per_customer", "coupon_discount_minor_per_customer", "marketing_cost_minor_per_customer",
)
ASSUMPTIONS = (
    "policy fixed before outcome observation without using post-assignment outcomes",
    "verifiable randomized assignment with support for both actions",
    "logged and target interventions have the same terms and timing",
    "no interference between customers and additive customer outcomes",
    "this fixed cohort is evaluated without dynamic budget reallocation",
    "policy search on this log does not establish independent effectiveness",
)


@dataclass(frozen=True)
class ExperimentEvidence:
    """Producer declarations for fields absent from the existing experiment log.

    For a Parquet path, log_sha256 must come from a trusted producer manifest.
    Matching the file checksum does not authenticate the manifest or prove randomization.
    """

    dataset_id: str
    experiment_id: str
    raw_database_sha256: str
    assignment_source: str | None
    assignment_design: str
    intervention_spec: str
    log_sha256: str | None = None


@dataclass(frozen=True)
class PolicyFixation:
    """Caller-supplied timing and post-outcome-use declarations, not audit proof."""

    fixed_at: object | None = None
    used_post_assignment_outcomes: bool | None = None


@dataclass(frozen=True)
class PolicyValueResult:
    summary: dict
    metrics: dict

    def to_dict(self) -> dict:
        return _json_values({"summary": self.summary, "metrics": self.metrics})


@dataclass(frozen=True)
class PolicyValueComparison:
    summary: dict
    metrics: dict

    def to_dict(self) -> dict:
        return _json_values({"summary": self.summary, "metrics": self.metrics})


def _json_values(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    return value


def _require(condition, message):
    if not condition:
        raise InputValidationError(f"policy_value: {message}")


def _sha(value, field):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"invalid {field}")


def _flag(value, field):
    _require(isinstance(value, (bool, np.bool_)), f"{field} requires a boolean")
    return bool(value)


def _minor(value, field, *, nullable=False):
    if nullable and pd.isna(value):
        return None
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)) and value >= 0,
             f"{field} requires nonnegative integer minor units")
    return int(value)


def _clock(value, field):
    return budget._clock(value, field)


def _read_log(source, evidence):
    if isinstance(source, pd.DataFrame):
        # Explicit internal contract: data shape is validated, file bytes are unavailable.
        return source.copy(deep=True), "unverified"
    _require(isinstance(source, (str, Path)), "experiment_log requires a DataFrame or Parquet path")
    _sha(evidence.log_sha256, "log_sha256")
    payload = Path(source).read_bytes()
    _require(sha256(payload).hexdigest() == evidence.log_sha256, "experiment_log checksum mismatch")
    return pd.read_parquet(BytesIO(payload)), "verified_against_supplied_manifest"


def _rewards(row, point, end):
    _require(_clock(row.assigned_at, "assigned_at") == point, "assignment must occur at as_of_at")
    _require(_clock(row.outcome_end_at, "outcome_end_at") == end, "outcome window must be 28 days")
    _require(_flag(row.eligible_at_assignment, "eligible_at_assignment"), "ineligible assignment row")
    action = _minor(row.treatment, "treatment")
    _require(action in (0, 1) and _minor(row.coupon_action, "coupon_action") == action,
             "coupon request must equal the original binary assignment")
    _require(budget._decimal(row.assignment_probability, "assignment_probability") == Fraction(1, 2),
             "logged assignment probability must equal 0.5")
    _require(_minor(row.additional_send_count, "additional_send_count") == 0, "repeat sends are unsupported")
    sent = _flag(row.actual_sent, "actual_sent")
    status = row.execution_status
    statuses = ("sent",) if sent else ("failed", "not_sent") if action else ("not_requested",)
    _require(isinstance(status, str) and status in statuses, "execution status contradicts assignment or send")
    if sent:
        _require(action == 1, "Control cannot send an experimental coupon")
        budget._text(row.coupon_id, "coupon_id")
        _require(_clock(row.sent_at, "sent_at") == point and _clock(row.expires_at, "expires_at") == end,
                 "experimental coupon must be sent at as_of_at and valid for 28 days")
        _require(budget._decimal(row.discount_rate, "discount_rate") == Fraction(1, 10),
                 "experimental coupon discount must be 10%")
    else:
        _require(all(pd.isna(x) for x in (row.coupon_id, row.sent_at, row.expires_at, row.discount_rate)),
                 "unsent customer has experimental coupon execution fields")

    observed = _flag(row.outcome_observed, "outcome_observed")
    no_order, no_session = (
        None if pd.isna(value) else _flag(value, name)
        for name, value in (("factual_no_order", row.factual_no_order),
                            ("factual_no_session", row.factual_no_session))
    )
    paid = _minor(row.factual_outcome_net_revenue_minor, "paid_minor", nullable=True)
    send_cost = _minor(row.send_cost_minor, "send_cost_minor", nullable=True)
    discount = _minor(row.outcome_redemption_discount_cost_minor, "coupon_discount_minor", nullable=True)
    _require(observed or all(value is None for value in (no_order, no_session, paid)),
             "censored row contains a finalized 28-day outcome")
    _require(sent or all(value in (None, 0) for value in (send_cost, discount)),
             "unsent customer has a realized experimental coupon charge")
    _require(no_order is not True or (paid in (None, 0) and discount in (None, 0)),
             "no-order row contains purchase amounts")

    # A redemption may be known before the observation window closes.
    # Keep the source log intact and withhold the full-window discount metric.
    finalized_discount = discount if observed else None
    marketing = None if send_cost is None or finalized_discount is None else send_cost + finalized_discount
    return action, observed, (None if no_order is None else int(not no_order), no_session, paid,
                              send_cost, finalized_discount, marketing)


def _prepare_log(source, evidence, allow_unverified_log):
    _require(isinstance(evidence, ExperimentEvidence), "ExperimentEvidence required")
    _flag(allow_unverified_log, "allow_unverified_log")
    _require(evidence.assignment_design == ASSIGNMENT_DESIGN, "unsupported assignment design")
    _require(evidence.intervention_spec == INTERVENTION_SPEC, "unsupported coupon intervention")
    for name in ("dataset_id", "experiment_id"):
        budget._text(getattr(evidence, name), name)
    _sha(evidence.raw_database_sha256, "raw_database_sha256")
    if evidence.log_sha256 is not None:
        _sha(evidence.log_sha256, "log_sha256")
    if evidence.assignment_source is not None:
        budget._text(evidence.assignment_source, "assignment_source")

    log, checksum_status = _read_log(source, evidence)
    optional = [name for name in ("raw_database_sha256", "is_censored") if name in log]
    required = (*LOG_COLUMNS, *optional)
    table = budget._table(log, required, "experiment_log")
    if "is_censored" in table:
        for observed, censored in zip(table.outcome_observed, table.is_censored):
            _require(_flag(censored, "is_censored") != _flag(observed, "outcome_observed"),
                     "is_censored contradicts outcome_observed")
    _require(len(table) > 0 and len(table) % 2 == 0, "fixed balanced randomization requires an even nonempty cohort")
    _require(table.dataset_id.eq(evidence.dataset_id).fillna(False).all()
             and table.experiment_id.eq(evidence.experiment_id).fillna(False).all(),
             "log dataset or experiment mismatch")
    if "raw_database_sha256" in table:
        _require(table.raw_database_sha256.eq(evidence.raw_database_sha256).fillna(False).all(),
                 "log raw SHA mismatch")
    for field in ("snapshot_id", "currency"):
        for value in table[field]:
            budget._text(value, field)
    table["amount_unit"] = [budget._decimal(value, "amount_unit", positive=True) for value in table.amount_unit]
    _require(all(table[column].nunique() == 1 for column in ("as_of_at", *MONEY_KEYS)),
             "mixed log as_of_at, currency or amount_unit")
    _require(not table.loc[table.coupon_id.notna(), "coupon_id"].duplicated().any(),
             "duplicate experimental coupon ID")
    table = table.sort_values(list(JOIN_KEYS)).reset_index(drop=True)
    point = table.as_of_at.iloc[0]
    end = point + pd.Timedelta(days=28)
    records = [_rewards(row, point, end) for row in table.itertuples(index=False)]
    _require(sum(action for action, _, _ in records) * 2 == len(records),
             "fixed balanced randomization requires equal Treatment and Control sizes")
    block = None
    if evidence.assignment_source is None:
        block = "assignment_source_unavailable"
    elif checksum_status == "unverified" and not allow_unverified_log:
        block = "log_checksum_unverified"
    provenance = {
        "dataset_id": evidence.dataset_id, "experiment_id": evidence.experiment_id,
        "raw_database_sha256": evidence.raw_database_sha256, "assignment_design": evidence.assignment_design,
        "assignment_source": evidence.assignment_source,
        "assignment_evidence": "producer_declared" if evidence.assignment_source else "unverified",
        "randomization_independently_verified": False, "log_checksum_status": checksum_status,
        "censor_flag_status": ("cross_checked" if "is_censored" in table else
                               "omitted_dataframe" if isinstance(source, pd.DataFrame) else "missing_from_file"),
        "declared_log_sha256": evidence.log_sha256, "intervention_spec": evidence.intervention_spec,
        "as_of_at": point.isoformat(), "outcome_end_at": end.isoformat(), "currency": table.currency.iloc[0],
        "amount_unit": str(table.amount_unit.iloc[0]),
    }
    return table, records, point, provenance, block


def _fixation_status(fixation, point):
    if fixation is None:
        return "unverified"
    _require(isinstance(fixation, PolicyFixation), "fixation requires PolicyFixation")
    if fixation.used_post_assignment_outcomes is not None:
        _require(not _flag(fixation.used_post_assignment_outcomes, "used_post_assignment_outcomes"),
                 "policy used post-assignment outcomes")
    if fixation.fixed_at is not None:
        _require(_clock(fixation.fixed_at, "fixed_at") <= point, "policy fixed after observation started")
    return ("caller_declared_pre_outcome" if fixation.fixed_at is not None
            and fixation.used_post_assignment_outcomes is False else "unverified")


def _plan(run, table, point, provenance):
    _require(isinstance(run, PolicyRun) and isinstance(run.summary, dict), "PolicyRun required")
    summary = deepcopy(run.summary)
    valid = _flag(summary.get("has_valid_solution"), "has_valid_solution")
    approved = _flag(summary.get("accepted_for_planning"), "accepted_for_planning")
    _require(valid or not approved, "invalid plan cannot be approved")
    try:
        config = summary["configuration"]
        candidate = PolicyCandidate(
            config["policy_id"], config["version"], config["description"], config["source"],
            SegmentationConfig(**config["segmentation"]), tuple(SegmentPolicy(**rule) for rule in config["policies"]),
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise InputValidationError("policy_value: invalid policy configuration") from exc
    _sha(summary.get("configuration_fingerprint"), "configuration_fingerprint")
    _require(candidate.configuration_fingerprint() == summary["configuration_fingerprint"],
             "configuration fingerprint mismatch")
    _require(candidate.policy_id == summary.get("policy_id") and candidate.version == summary.get("policy_version"),
             "policy ID or version mismatch")
    _require(summary.get("dataset_id") == provenance["dataset_id"]
             and _clock(summary.get("as_of_at"), "policy.as_of_at") == point, "policy cohort mismatch")
    columns = (*IDENTITY, *MONEY_KEYS, "selected", "planned_sent_at", "budget_candidate", "expected_cost", "algorithm")
    frame = budget._table(run.decisions, columns, "policy_decisions").sort_values(list(JOIN_KEYS)).reset_index(drop=True)
    algorithm = summary.get("algorithm")
    allocation = summary.get("allocation_metadata")
    _require(algorithm in ("greedy", "milp", "hybrid") and frame.algorithm.eq(algorithm).fillna(False).all(),
             "policy algorithm differs between summary and decisions")
    _require(isinstance(allocation, dict) and ("hybrid" in allocation) == (algorithm == "hybrid"),
             "policy algorithm conflicts with allocation metadata")
    if algorithm == "hybrid":
        _require(not approved, "hybrid cannot be accepted_for_planning")
        hybrid = allocation["hybrid"]
        quality = hybrid.get("quality") if isinstance(hybrid, dict) else None
        status = summary.get("status")
        _require(isinstance(quality, dict) and quality.get("status") == status
                 and status in ("proven_optimal", "epsilon_certified", "feasible_uncertified", "unavailable")
                 and valid == (status != "unavailable"), "hybrid quality status conflicts with policy summary")
    _require(len(frame) == len(table) and frame[list(JOIN_KEYS)].eq(table[list(JOIN_KEYS)]).all().all(),
             "policy and log roster keys differ")
    _require(frame.snapshot_id.eq(table.snapshot_id).fillna(False).all(), "policy and log snapshot_id differ")
    _require(frame.raw_database_sha256.eq(provenance["raw_database_sha256"]).fillna(False).all(), "policy raw SHA mismatch")
    _require(frame.currency.eq(provenance["currency"]).fillna(False).all()
             and all(budget._decimal(unit, "policy.amount_unit", positive=True) == table.amount_unit.iloc[0]
                     for unit in frame.amount_unit), "policy currency or amount_unit mismatch")
    _require(_minor(summary.get("population_count"), "population_count") == len(frame), "policy population count mismatch")

    fingerprint = summary.get("decision_fingerprint")
    actions = None
    if valid:
        _sha(fingerprint, "decision_fingerprint")
        actions = [_minor(value, "selected") for value in frame.selected]
        _require(all(action in (0, 1) for action in actions), "selected requires a binary action")
        for row, action in zip(frame.itertuples(index=False), actions):
            is_candidate = _flag(row.budget_candidate, "budget_candidate")
            if action:
                _require(is_candidate and _clock(row.planned_sent_at, "planned_sent_at") == point,
                         "selected actions must be candidates scheduled at as_of_at")
            else:
                _require(pd.isna(row.planned_sent_at), "unselected row has planned_sent_at")
        expected = _fingerprint(frame.sort_values(list(IDENTITY))[
            [*IDENTITY, "selected", "planned_sent_at"]
        ].to_dict("records"))
        _require(expected == fingerprint, "decision fingerprint mismatch")
        _require(_flag(summary.get("budget_compliant"), "budget_compliant"), "valid policy must comply with budget")
        total, remaining, limit = [
            Fraction(budget._decimal(summary.get(name), name))
            for name in ("total_expected_cost", "remaining_budget", "budget")
        ]
        computed = sum(Fraction(budget._decimal(value, "expected_cost"))
                       for value, action in zip(frame.expected_cost, actions) if action)
        _require(computed == total and total + remaining == limit, "policy expected cost or budget mismatch")
        _require(_minor(summary.get("selected_customer_count"), "selected_customer_count") == sum(actions),
                 "policy selected count mismatch")
    else:
        _require(fingerprint is None, "invalid plan must not have a decision fingerprint")
    return {
        "policy_id": candidate.policy_id, "policy_version": candidate.version,
        "configuration_fingerprint": summary["configuration_fingerprint"], "decision_fingerprint": fingerprint,
        "has_valid_solution": valid, "accepted_for_planning": approved,
        "budget_compliant": bool(summary["budget_compliant"]) if valid else None,
        "planned_expected_cost": str(summary["total_expected_cost"]) if valid else None,
    }, actions


def _evaluate(run, prepared, fixation, allow_unapproved, warn_ess_below):
    table, records, point, provenance, evidence_block = prepared
    _flag(allow_unapproved, "allow_unapproved")
    if warn_ess_below is not None:
        warn_ess_below = _minor(warn_ess_below, "warn_ess_below")
    fixation_status = _fixation_status(fixation, point)
    plan, actions = _plan(run, table, point, provenance)
    total = len(records)
    observed = sum(complete for _, complete, _ in records)
    # The producer contract and every logged probability have already been
    # checked. The only supported reciprocal propensity is exactly two.
    weights = None if actions is None else [2 if action == logged else 0
                                            for action, (logged, _, _) in zip(actions, records)]
    matches = None if weights is None else sum(weight > 0 for weight in weights)
    weight_sum = None if weights is None else sum(weights)
    weight_squared = None if weights is None else sum(weight * weight for weight in weights)
    ess = None if weights is None else Fraction(weight_sum ** 2, weight_squared) if weight_squared else Fraction(0)
    block = ("invalid_plan" if actions is None else "unapproved_plan"
             if not plan["accepted_for_planning"] and not allow_unapproved else evidence_block)
    if block is None and matches == 0:
        block = "no_action_matches"
    warnings = []
    if ess is not None and warn_ess_below is not None and ess < warn_ess_below:
        warnings.append("effective_sample_size_below_requested_threshold")
    if fixation_status == "unverified":
        warnings.append("policy_fixation_unverified")

    diagnostics = {
        "population_count": total, "outcome_observed_count": observed, "censored_count": total - observed,
        "send_requested_count": None if actions is None else sum(actions),
        "no_send_requested_count": None if actions is None else total - sum(actions),
        "matched_count": matches, "match_rate": None if matches is None else Fraction(matches, total),
        "weight_sum": weight_sum, "maximum_weight": None if weights is None else max(weights),
        "effective_sample_size": ess,
    }
    metrics, contributions = {}, {}
    for index, name in enumerate(METRICS):
        values = [reward[index] for _, _, reward in records]
        metric_observed = sum(value is not None for value in values)
        reason = block
        if reason is None and metric_observed != total:
            reason = ("censored_outcomes" if observed != total and name != "send_cost_minor_per_customer"
                      else "missing_metric_observations")
        if reason is None:
            contributions[name] = tuple(weight * value for weight, value in zip(weights, values))
        metrics[name] = {
            **diagnostics, "metric_observed_count": metric_observed, "estimable": reason is None,
            "hold_reason": reason,
            "estimate": Fraction(sum(contributions[name]), total) if reason is None else None,
        }
    estimable_count = sum(metric["estimable"] for metric in metrics.values())
    summary = {
        **plan, **provenance, **diagnostics, "policy_fixation": fixation_status,
        "readiness": "held" if estimable_count == 0 else "partially_estimable" if estimable_count < len(METRICS)
        else "point_estimates_available",
        "exploratory": (not plan["accepted_for_planning"] or provenance["log_checksum_status"] == "unverified"
                        or fixation_status == "unverified"),
        "warnings": warnings, "method": "ips", "causal_interpretation": "unverified",
        "dispatch_authorized": False, "assumptions": ASSUMPTIONS,
    }
    return PolicyValueResult(summary, metrics), contributions


def evaluate_policy_value(
    run: PolicyRun, experiment_log: pd.DataFrame | str | Path, *, evidence: ExperimentEvidence,
    fixation: PolicyFixation | None = None, allow_unapproved: bool = False,
    allow_unverified_log: bool = False, warn_ess_below: int | None = None,
) -> PolicyValueResult:
    """Evaluate a Phase A plan using initial assignment, with the full-cohort denominator.

    DataFrame point estimates require explicit exploratory opt-in. Estimates and
    ESS are exact rational values and serialize as numerator/denominator pairs.
    """
    prepared = _prepare_log(experiment_log, evidence, allow_unverified_log)
    result, _ = _evaluate(run, prepared, fixation, allow_unapproved, warn_ess_below)
    return result


def compare_policy_values(
    left: PolicyRun, right: PolicyRun, experiment_log: pd.DataFrame | str | Path, *,
    evidence: ExperimentEvidence, left_fixation: PolicyFixation | None = None,
    right_fixation: PolicyFixation | None = None, allow_unapproved: bool = False,
    allow_unverified_log: bool = False, warn_ess_below: int | None = None,
) -> PolicyValueComparison:
    """Compute left minus right from aligned customer contributions on one shared log."""
    prepared = _prepare_log(experiment_log, evidence, allow_unverified_log)
    left_value, a = _evaluate(left, prepared, left_fixation, allow_unapproved, warn_ess_below)
    right_value, b = _evaluate(right, prepared, right_fixation, allow_unapproved, warn_ess_below)
    total = left_value.summary["population_count"]
    metrics = {}
    for name in METRICS:
        estimable = name in a and name in b
        metrics[name] = {
            "population_count": total, "estimable": estimable,
            "left_hold_reason": left_value.metrics[name]["hold_reason"],
            "right_hold_reason": right_value.metrics[name]["hold_reason"],
            "left_matched_count": left_value.summary["matched_count"],
            "right_matched_count": right_value.summary["matched_count"],
            "left_effective_sample_size": left_value.summary["effective_sample_size"],
            "right_effective_sample_size": right_value.summary["effective_sample_size"],
            "difference": Fraction(sum(x - y for x, y in zip(a[name], b[name])), total) if estimable else None,
        }
    return PolicyValueComparison({
        "direction": "left_minus_right", "comparison_scope": "same_log_paired_customer_contributions",
        "causal_interpretation": "unverified", "dispatch_authorized": False,
        "left": left_value.summary, "right": right_value.summary,
    }, metrics)
