"""Exact IPS expectations on synthetic assignments and Phase A plans."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json

import pandas as pd
import pytest

from final_strategy.budget import Budget
from final_strategy.policy_search import PolicyRun, run_policy_search
from final_strategy.policy_value import (
    ASSIGNMENT_DESIGN, INTERVENTION_SPEC, ExperimentEvidence, PolicyFixation,
    compare_policy_values, evaluate_policy_value,
)
from final_strategy.tests.test_policy_search import BASE, RECORDS, T0, _inputs, _policy
from final_strategy.validation import InputValidationError


@pytest.fixture(scope="module")
def runs():
    disabled = _policy("none", lambda rule: replace(rule, allow_candidate=False))
    later = _policy("later", lambda rule: replace(rule, send_delay="1 day"))
    plans = run_policy_search(candidates=[BASE, disabled, later], **_inputs()).runs
    all_positive = [(customer, churn, abs(uplift), clv) for customer, churn, uplift, clv in RECORDS]
    options = _inputs(all_positive)
    options["budget"] = Budget(1, "SOURCE_UNSPECIFIED", ".01")
    plans["all"] = run_policy_search(candidates=[replace(BASE, policy_id="all")], **options).runs["all"]
    return plans


@pytest.fixture
def inputs(runs):
    rows = []
    # 001 and 004 were assigned treatment; 004 did not actually receive a coupon.
    for i, row in enumerate(runs["base"].decisions.sort_values("customer_id").itertuples()):
        action = int(i in (0, 3))
        sent = i == 0
        rows.append(dict(
            customer_id=row.customer_id, dataset_id=row.dataset_id, snapshot_id=row.snapshot_id,
            as_of_at=T0, assigned_at=T0, currency=row.currency, amount_unit=row.amount_unit,
            experiment_id="fixture", treatment=action, assignment_probability=.5,
            coupon_action=action, actual_sent=sent, coupon_id="single" if sent else None,
            execution_status="sent" if sent else "failed" if action else "not_requested",
            sent_at=T0 if sent else pd.NaT, expires_at=T0 + pd.Timedelta(days=28) if sent else pd.NaT,
            discount_rate=.1 if sent else None, additional_send_count=0, eligible_at_assignment=True,
            outcome_end_at=T0 + pd.Timedelta(days=28), outcome_observed=True,
            factual_no_order=i == 2, factual_no_session=i == 2,
            factual_outcome_net_revenue_minor=[90, 50, 0, 40][i],
            send_cost_minor=5 if sent else 0, outcome_redemption_discount_cost_minor=10 if sent else 0,
        ))
    log = pd.DataFrame(rows)
    for column in ("factual_outcome_net_revenue_minor", "send_cost_minor", "outcome_redemption_discount_cost_minor"):
        log[column] = log[column].astype("Int64")
    for column in ("factual_no_order", "factual_no_session"):
        log[column] = log[column].astype("boolean")
    evidence = ExperimentEvidence(rows[0]["dataset_id"], "fixture", "a" * 64, "synthetic fixed assignment",
                                  ASSIGNMENT_DESIGN, INTERVENTION_SPEC)
    return log, evidence


def _evaluate(run, inputs, **options):
    log, evidence = inputs
    return evaluate_policy_value(run, log, evidence=evidence, **({"allow_unverified_log": True} | options))


def test_full_roster_denominator_and_no_clipping(runs, inputs):
    estimates = {}
    for plan in ("base", "none", "all"):
        estimates[plan] = _evaluate(runs[plan], inputs)
    part = estimates["base"]
    assert part.summary["population_count"] == 4
    assert part.summary["send_requested_count"] == 2
    assert part.summary["matched_count"] == 4
    assert part.summary["weight_sum"] == 8
    assert part.summary["maximum_weight"] == 2
    assert part.summary["effective_sample_size"] == 4
    assert part.metrics["purchase_rate_28d"]["estimate"] == Fraction(3, 2)
    assert part.metrics["no_session_rate_28d"]["estimate"] == Fraction(1, 2)
    assert estimates["none"].metrics["purchase_rate_28d"]["estimate"] == Fraction(1, 2)
    assert estimates["all"].metrics["purchase_rate_28d"]["estimate"] == 1
    assert estimates["all"].summary["effective_sample_size"] == 2
    assert estimates["all"].summary["match_rate"] == Fraction(1, 2)


def test_money_and_failed_delivery_keep_original_assignment(runs, inputs):
    result = _evaluate(runs["all"], inputs)
    # Both assigned treatments match this policy, including customer 004 whose delivery failed.
    assert result.metrics["paid_minor_per_customer"]["estimate"] == 65
    assert result.metrics["send_cost_minor_per_customer"]["estimate"] == Fraction(5, 2)
    assert result.metrics["coupon_discount_minor_per_customer"]["estimate"] == 5
    assert result.metrics["marketing_cost_minor_per_customer"]["estimate"] == Fraction(15, 2)
    assert result.summary["currency"] == "SOURCE_UNSPECIFIED"
    assert Fraction(result.summary["amount_unit"]) == Fraction(1, 100)
    assert result.summary["causal_interpretation"] == "unverified"
    assert result.summary["dispatch_authorized"] is False


def test_minor_precision_survives_large_totals_and_json(runs, inputs):
    log, evidence = inputs
    log.loc[0, "factual_outcome_net_revenue_minor"] = 2**62 + 1
    log.loc[3, "factual_outcome_net_revenue_minor"] = 2**62 + 1
    result = _evaluate(runs["all"], (log, evidence))
    assert result.metrics["paid_minor_per_customer"]["estimate"] == 2**62 + 1
    serialized = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert serialized["metrics"]["paid_minor_per_customer"]["estimate"] == {
        "numerator": 2**62 + 1, "denominator": 1,
    }
    assert serialized["metrics"]["send_cost_minor_per_customer"]["estimate"] == {
        "numerator": 5, "denominator": 2,
    }


def test_same_log_paired_differences_and_order_independence(runs, inputs):
    log, evidence = inputs
    kwargs = dict(evidence=evidence, allow_unverified_log=True)
    before = log.copy(deep=True)
    original = runs["base"].decisions.copy(deep=True)
    summary = deepcopy(runs["base"].summary)
    reversed_plan = PolicyRun(original.iloc[::-1], summary)
    comparison = compare_policy_values(reversed_plan, runs["none"], log.iloc[::-1], **kwargs)
    same = compare_policy_values(runs["base"], reversed_plan, log, **kwargs)
    reverse = compare_policy_values(runs["none"], runs["base"], log, **kwargs)
    assert comparison.metrics["purchase_rate_28d"]["difference"] == 1
    assert comparison.metrics["paid_minor_per_customer"]["difference"] == 65
    assert reverse.metrics["paid_minor_per_customer"]["difference"] == -65
    assert all(item["difference"] == 0 for item in same.metrics.values())
    assert comparison.metrics["paid_minor_per_customer"]["right_matched_count"] == 2
    assert comparison.summary["direction"] == "left_minus_right"
    assert json.dumps(comparison.to_dict(), allow_nan=False)
    pd.testing.assert_frame_equal(log, before)
    pd.testing.assert_frame_equal(runs["base"].decisions, original)
    assert runs["base"].summary == summary


def test_no_matches_and_low_ess_are_distinct(runs, inputs):
    log, evidence = inputs
    override = log.copy(deep=True)
    other_rows = override.iloc[[1, 0, 3, 2]].reset_index(drop=True)
    action_columns = ["treatment", "coupon_action", "actual_sent", "coupon_id", "execution_status", "sent_at",
                      "expires_at", "discount_rate", "send_cost_minor", "outcome_redemption_discount_cost_minor"]
    override[action_columns] = other_rows[action_columns]
    # Put the redeemed coupon on an observed buyer, not on the original no-order row.
    override.loc[1, "factual_no_order"] = False
    held = _evaluate(runs["base"], (override, evidence), warn_ess_below=3)
    assert held.summary["matched_count"] == held.summary["effective_sample_size"] == 0
    assert held.metrics["purchase_rate_28d"]["hold_reason"] == "no_action_matches"
    assert held.metrics["purchase_rate_28d"]["estimate"] is None
    low = _evaluate(runs["none"], inputs, warn_ess_below=3)
    assert low.summary["effective_sample_size"] == 2
    assert "effective_sample_size_below_requested_threshold" in low.summary["warnings"]
    assert low.metrics["purchase_rate_28d"]["estimable"]


def test_censored_row_preserves_partial_discount_and_estimates_settled_send_cost(runs, inputs):
    log, evidence = inputs
    log.loc[0, "outcome_observed"] = False
    log.loc[0, ["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]] = pd.NA
    result = _evaluate(runs["base"], (log, evidence))
    assert log.loc[0, "outcome_redemption_discount_cost_minor"] == 10
    assert result.summary["censored_count"] == 1
    assert result.summary["readiness"] == "partially_estimable"
    assert result.metrics["purchase_rate_28d"]["estimate"] is None
    assert result.metrics["purchase_rate_28d"]["hold_reason"] == "censored_outcomes"
    assert result.metrics["coupon_discount_minor_per_customer"]["estimate"] is None
    assert result.metrics["marketing_cost_minor_per_customer"]["estimate"] is None
    assert result.metrics["send_cost_minor_per_customer"]["estimate"] == Fraction(5, 2)


def test_missing_metric_for_nonmatching_customer_still_blocks_full_cohort(runs, inputs):
    log, evidence = inputs
    log.loc[0, "factual_outcome_net_revenue_minor"] = pd.NA
    result = _evaluate(runs["none"], (log, evidence))
    assert result.metrics["paid_minor_per_customer"]["hold_reason"] == "missing_metric_observations"
    assert result.metrics["paid_minor_per_customer"]["metric_observed_count"] == 3
    assert result.metrics["purchase_rate_28d"]["estimable"]


def test_evidence_and_verified_file_bytes_are_separate(runs, inputs, tmp_path):
    log, evidence = inputs
    assert _evaluate(runs["base"], inputs, allow_unverified_log=False).metrics[
        "purchase_rate_28d"]["hold_reason"] == "log_checksum_unverified"
    assert _evaluate(runs["base"], (log, replace(evidence, assignment_source=None))).metrics[
        "purchase_rate_28d"]["hold_reason"] == "assignment_source_unavailable"
    path = tmp_path / "experiment_log.parquet"
    log.to_parquet(path, index=False)
    declared = replace(evidence, log_sha256=sha256(path.read_bytes()).hexdigest())
    result = evaluate_policy_value(runs["base"], path, evidence=declared, fixation=PolicyFixation(T0, False))
    assert result.summary["log_checksum_status"] == "verified_against_supplied_manifest"
    assert result.summary["assignment_evidence"] == "producer_declared"
    assert result.summary["randomization_independently_verified"] is False
    assert result.summary["policy_fixation"] == "caller_declared_pre_outcome"
    assert result.metrics["purchase_rate_28d"]["estimate"] == Fraction(3, 2)
    assert _evaluate(runs["base"], (log, declared)).summary["log_checksum_status"] == "unverified"
    with pytest.raises(InputValidationError, match="checksum mismatch"):
        evaluate_policy_value(runs["base"], path, evidence=replace(declared, log_sha256="f" * 64))


@pytest.mark.parametrize(("field", "value", "message"), [
    ("assignment_probability", .75, "probability must equal"),
    ("assignment_probability", float("inf"), "assignment_probability"),
    ("assigned_at", T0 + pd.Timedelta(seconds=1), "assignment must occur"),
    ("outcome_end_at", T0 + pd.Timedelta(days=30), "28 days"),
    ("discount_rate", .2, "discount must be 10%"),
    ("expires_at", T0 + pd.Timedelta(days=27), "valid for 28 days"),
    ("additional_send_count", 1, "repeat sends"),
    ("snapshot_id", "different", "snapshot_id differ"),
    ("dataset_id", "different", "dataset or experiment"),
    ("currency", "KRW", "mixed log"),
    ("amount_unit", ".02", "mixed log"),
    ("raw_database_sha256", "z" * 64, "log raw SHA mismatch"),
    ("execution_status", "failed", "execution status contradicts"),
    ("factual_outcome_net_revenue_minor", -1, "integer minor units"),
    ("factual_outcome_net_revenue_minor", .51, "integer minor units"),
    ("outcome_observed", False, "censored row contains"),
])
def test_rejects_invalid_log_values(runs, inputs, field, value, message):
    log, evidence = inputs
    if field == "raw_database_sha256":
        log[field] = evidence.raw_database_sha256
    log[field] = log[field].astype(object)
    log.loc[0, field] = value
    with pytest.raises(InputValidationError, match=message):
        _evaluate(runs["base"], (log, evidence))


def test_rejects_incomplete_duplicate_and_unequal_rosters(runs, inputs):
    log, evidence = inputs
    with pytest.raises(InputValidationError, match="even nonempty"):
        _evaluate(runs["base"], (log.iloc[:-1], evidence))
    with pytest.raises(InputValidationError, match="duplicate customer"):
        _evaluate(runs["base"], (pd.concat([log.iloc[:3], log.iloc[:1]]), evidence))
    with pytest.raises(InputValidationError, match="roster keys differ"):
        _evaluate(runs["base"], (log.iloc[:2], evidence))
    log.loc[1, ["treatment", "coupon_action", "execution_status"]] = [1, 1, "not_sent"]
    with pytest.raises(InputValidationError, match="equal Treatment and Control"):
        _evaluate(runs["base"], inputs)


@pytest.mark.parametrize("field", ["assignment_design", "intervention_spec"])
def test_unsupported_producer_contract(runs, inputs, field):
    log, evidence = inputs
    with pytest.raises(InputValidationError, match="unsupported"):
        _evaluate(runs["base"], (log, replace(evidence, **{field: "other"})))


def test_unapproved_and_invalid_plans_are_not_converted_to_no_send(runs, inputs):
    base = runs["base"]
    unapproved = replace(base, summary=base.summary | {"accepted_for_planning": False})
    held = _evaluate(unapproved, inputs)
    assert held.metrics["purchase_rate_28d"]["hold_reason"] == "unapproved_plan"
    exploratory = _evaluate(unapproved, inputs, allow_unapproved=True)
    assert exploratory.summary["accepted_for_planning"] is False
    assert exploratory.summary["exploratory"]
    assert exploratory.summary["dispatch_authorized"] is False
    assert exploratory.metrics["purchase_rate_28d"]["estimable"]
    decisions = base.decisions.copy(deep=True)
    decisions.loc[decisions.budget_candidate, "selected"] = pd.NA
    invalid = PolicyRun(decisions, base.summary | {
        "has_valid_solution": False, "accepted_for_planning": False, "decision_fingerprint": None,
    })
    result = _evaluate(invalid, inputs, allow_unapproved=True)
    assert result.summary["no_send_requested_count"] is None
    assert result.metrics["purchase_rate_28d"]["hold_reason"] == "invalid_plan"
    assert _evaluate(runs["none"], inputs).summary["send_requested_count"] == 0


def test_rejects_incompatible_phase_a_plan_and_tampering(runs, inputs):
    with pytest.raises(InputValidationError, match="scheduled at as_of_at"):
        _evaluate(runs["later"], inputs)
    base = runs["base"]
    for field in ("decision_fingerprint", "configuration_fingerprint"):
        with pytest.raises(InputValidationError, match="fingerprint mismatch"):
            _evaluate(replace(base, summary=base.summary | {field: "b" * 64}), inputs)
    with pytest.raises(InputValidationError, match="expected cost or budget mismatch"):
        _evaluate(replace(base, summary=base.summary | {"remaining_budget": "1"}), inputs)
    for field, value, message in [
        ("snapshot_id", "changed", "snapshot_id differ"),
        ("raw_database_sha256", "b" * 64, "policy raw SHA mismatch"),
        ("currency", "KRW", "policy currency or amount_unit"),
    ]:
        frame = base.decisions.copy(deep=True)
        frame[field] = value
        with pytest.raises(InputValidationError, match=message):
            _evaluate(replace(base, decisions=frame), inputs)


@pytest.mark.parametrize(("fixation", "message"), [
    (PolicyFixation(T0 + pd.Timedelta(seconds=1), False), "fixed after"),
    (PolicyFixation(T0, True), "used post-assignment"),
])
def test_rejects_declared_outcome_leakage(runs, inputs, fixation, message):
    with pytest.raises(InputValidationError, match=message):
        _evaluate(runs["base"], inputs, fixation=fixation)


@pytest.fixture(scope="module")
def hybrid_run():
    return run_policy_search(candidates=[BASE], **(_inputs() | {"algorithm": "hybrid"})).runs["base"]


def test_hybrid_requires_explicit_exploration(hybrid_run, inputs):
    assert hybrid_run.summary["status"] == "proven_optimal"
    held = _evaluate(hybrid_run, inputs)
    assert held.metrics["purchase_rate_28d"]["hold_reason"] == "unapproved_plan"
    value = _evaluate(hybrid_run, inputs, allow_unapproved=True)
    assert value.metrics["purchase_rate_28d"]["estimate"] == Fraction(3, 2)
    assert value.summary["exploratory"] and not value.summary["accepted_for_planning"]
    assert not held.summary["dispatch_authorized"] and not value.summary["dispatch_authorized"]


@pytest.mark.parametrize("allow_unapproved", [False, True])
def test_hybrid_rejects_contradictory_approval(hybrid_run, inputs, allow_unapproved):
    changed = replace(hybrid_run, summary=hybrid_run.summary | {"accepted_for_planning": True})
    with pytest.raises(InputValidationError, match="hybrid cannot be accepted"):
        _evaluate(changed, inputs, allow_unapproved=allow_unapproved)


@pytest.mark.parametrize("field", ["algorithm", "allocation_metadata", "status"])
def test_hybrid_rejects_conflicting_metadata(hybrid_run, inputs, field):
    summary = deepcopy(hybrid_run.summary)
    if field == "allocation_metadata":
        del summary[field]["hybrid"]
    else:
        summary[field] = "greedy" if field == "algorithm" else "unavailable"
    with pytest.raises(InputValidationError, match="algorithm|quality status"):
        _evaluate(replace(hybrid_run, summary=summary), inputs, allow_unapproved=True)


@pytest.mark.parametrize("algorithm", ["greedy", "milp"])
def test_approved_budget_algorithms(algorithm, inputs):
    plan = run_policy_search(candidates=[BASE], **(_inputs() | {"algorithm": algorithm})).runs["base"]
    result = _evaluate(plan, inputs)
    assert result.summary["accepted_for_planning"] and not result.summary["dispatch_authorized"]
    assert result.metrics["purchase_rate_28d"]["estimable"]


def test_censor_evidence_sources(runs, inputs, tmp_path):
    log, evidence = inputs
    assert _evaluate(runs["base"], inputs).summary["censor_flag_status"] == "omitted_dataframe"
    path = tmp_path / "minimal.parquet"
    log.to_parquet(path, index=False)
    declared = replace(evidence, log_sha256=sha256(path.read_bytes()).hexdigest())
    assert evaluate_policy_value(runs["base"], path, evidence=declared).summary["censor_flag_status"] == "missing_from_file"
    log = log.assign(is_censored=False)
    assert _evaluate(runs["base"], (log, evidence)).summary["censor_flag_status"] == "cross_checked"
    log.loc[0, ["is_censored", "outcome_observed"]] = [True, False]
    log.loc[0, ["factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"]] = pd.NA
    before = log.copy(deep=True)
    result = _evaluate(runs["base"], (log, evidence))
    assert result.summary["censored_count"] == 1 and result.summary["population_count"] == 4
    assert result.metrics["purchase_rate_28d"]["estimate"] is None
    assert result.metrics["send_cost_minor_per_customer"]["estimate"] == Fraction(5, 2)
    pd.testing.assert_frame_equal(log, before)


@pytest.mark.parametrize("observed,censored", [(True, True), (False, False), (True, "False"),
                                               (True, 0), (True, None), (True, pd.NA)])
def test_rejects_one_invalid_censor_flag_in_either_api(runs, inputs, observed, censored):
    log, evidence = inputs
    log = log.assign(is_censored=pd.Series(False, index=log.index, dtype=object))
    log.loc[1, "outcome_observed"] = observed
    log.at[1, "is_censored"] = censored
    before = log.copy(deep=True)
    with pytest.raises(InputValidationError, match="is_censored"):
        _evaluate(runs["base"], (log, evidence))
    with pytest.raises(InputValidationError, match="is_censored"):
        compare_policy_values(runs["base"], runs["none"], log, evidence=evidence, allow_unverified_log=True)
    pd.testing.assert_frame_equal(log, before)
