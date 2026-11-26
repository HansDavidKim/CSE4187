"""Synthetic ledger checks only. These fixtures do not model treatment effects."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest

from final_strategy.experiment_power import PowerScenario, plan_sample_size
from final_strategy.experiment_spec import ExperimentSpec
from final_strategy.experiment_validation import CONTRACT_VERSION, validate_experiment_run
from final_strategy.validation import InputValidationError
from simulate.v4.features import BEHAVIOR_FEATURE_VERSION, EXPANDED_FEATURES, EXPANDED_FEATURE_VERSION


T0 = pd.Timestamp("2026-01-04T15:00:00Z")
END = T0 + timedelta(days=28)
ARMS = ("A", "B", "C", "D1", "D2")
IDS = ("001", "NA", "003", "004", "005")
SHA_PRE, SHA_POST = "a" * 64, "b" * 64


@pytest.fixture
def run():
    spec = ExperimentSpec("dummy", "v1", (1,) * 5)
    manifest = {
        "contract_version": CONTRACT_VERSION, "experiment_id": "dummy", "version": "v1",
        "evaluation_dataset_id": "eval", "as_of_at": T0, "population_frozen_at": T0,
        "observation_start_at": T0 - timedelta(days=28), "observation_end_at": END,
        "outcome_start_rule": "as_of_at", "currency": "KRW", "amount_unit": Decimal("1"),
        "eligibility_policy_version": "active28-v1", "coupon_policy_version": "coupon28-v1",
        "assignment_algorithm": "declared-fixture", "assignment_version": "v1",
        "assignment_seed": np.int64(42), "assignment_method": "independent",
        "feature_raw_sha256": SHA_PRE, "result_raw_sha256": SHA_POST,
        "ledger_sha256": {name: "c" * 64 for name in
                          ("population", "assignments", "decisions", "executions", "outcomes")},
        "budget_per_customer": "10.0",
        "arms": {arm: {
            "strategy_version": "full-v1" if arm.startswith("D") else arm,
            "model_version": "model-v1", "score_definition_id": "score-v1",
            "cost_model_version": "cost-v1", "solver_kind": {"D1": "greedy", "D2": "milp"}.get(arm, "none"),
            "solver_version": "v1", "decision_artifact_id": arm + "-artifact",
            "budget": 0 if arm == "A" else "10.0",
        } for arm in ARMS},
    }
    tables = {name: [] for name in manifest["ledger_sha256"]}
    for uid, arm in zip(IDS + ("excluded",), ARMS + (None,)):
        context = {"evaluation_dataset_id": "eval", "customer_id": uid,
                   "snapshot_id": f"s-{uid}", "as_of_at": T0,
                   "raw_database_sha256": SHA_PRE, "currency": "KRW", "amount_unit": 1}
        tables["population"].append(dict(context, eligible=arm is not None, eligibility_at=T0,
                                         exclusion_reason=None if arm else "ineligible"))
        if arm is None:
            continue
        context["arm_id"] = arm
        tables["assignments"].append(dict(context, assigned_at=T0, assignment_probability=np.float32(.2)))
        requested = arm != "A"
        planned = T0 + timedelta(hours=1) if requested else None
        tables["decisions"].append(dict(context, decision_at=T0, candidate=requested, selected=requested,
                                        send_requested=requested, planned_sent_at=planned,
                                        expected_cost="8.000000001" if requested else None))
        context["raw_database_sha256"] = SHA_POST
        sent = arm in ("D1", "D2")
        status = {"A": "not_requested", "B": "failed", "C": "not_sent"}.get(arm, "sent")
        tables["executions"].append(dict(context, actual_sent=sent, sent_at=planned if sent else None,
                                         coupon_id=f"c-{uid}" if sent else None,
                                         expires_at=planned + timedelta(days=28) if sent else None,
                                         discount_rate="0.1" if sent else None, execution_status=status,
                                         failure_reason="declared reason" if arm in ("B", "C") else None,
                                         send_cost_minor=4 if sent else 0))
        observed = arm != "C"
        tables["outcomes"].append(dict(context, outcome_start_at=T0, outcome_end_at=END,
                                       outcome_observed=observed, censoring_reason=None if observed else "data_gap",
                                       purchase_occurred=arm == "D2" if observed else None,
                                       session_occurred=arm in ("B", "D2") if observed else None,
                                       paid_minor=(25 if arm == "D2" else 0) if observed else None,
                                       coupon_discount_minor=(7 if arm == "D2" else 0) if observed else None,
                                       send_cost_minor=4 if sent else 0))
    return spec, manifest, {name: pd.DataFrame(rows, dtype=object) for name, rows in tables.items()}


def validate(run, **kwargs):
    spec, manifest, tables = run
    return validate_experiment_run(spec, manifest, **tables, **kwargs)


def test_itt_counts_budget_censoring_and_json(run):
    result = validate(run)
    assert result.evaluation_population == 6
    assert result.eligible_population == 5
    assert {arm: s["itt"] for arm, s in result.by_arm.items()} == dict.fromkeys(ARMS, 1)
    a, b, c, d1, d2 = (result.by_arm[arm] for arm in ARMS)
    assert (a["requested"], a["sent"], a["observed"], a["not_requested"]) == (0, 0, 1, 1)
    assert (b["requested"], b["sent"], b["failed"], b["observed"]) == (1, 0, 1, 1)
    assert (c["requested"], c["not_sent"], c["censored"]) == (1, 1, 1)
    assert c["realized_spend_minor"] is None and c["realized_budget_status"] == "unverified_incomplete"
    assert d1["sent"] == 1 and d1["realized_spend_minor"] == 4
    assert d2["selected_expected_cost"] == "8.000000001"
    assert d2["plan_within_budget"] and d2["realized_budget_status"] == "over_budget"
    assert d2["realized_spend_minor"] == 11
    report = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert report["data_integrity"] == "passed"
    assert report["design_status"] == "planning_unapproved"
    assert set(report["evidence_status"].values()) == {"unverified"}


@pytest.mark.parametrize("ledger,row,field,value,error", [
    ("population", 5, "customer_id", "001", "duplicate"),
    ("population", 5, "exclusion_reason", None, "exclusion_reason"),
    ("population", 0, "customer_id", 1, "requires nonempty strings"),
    ("population", 0, "eligibility_at", T0 + timedelta(seconds=1), "eligibility"),
    ("assignments", 0, "snapshot_id", "other", "eligible roster"),
    ("assignments", 0, "arm_id", None, "arm_id"),
    ("assignments", 0, "arm_id", "Treatment", "arm_id"),
    ("assignments", 0, "assignment_probability", .3, "assignment_probability"),
    ("assignments", 0, "assignment_probability", float("nan"), "assignment_probability"),
    ("assignments", 0, "assignment_probability", True, "assignment_probability"),
    ("assignments", 0, "assigned_at", "2026-01-04T15:00:00", "timezone-aware"),
    ("assignments", 0, "assigned_at", T0 - timedelta(seconds=1), "assigned_at"),
    ("decisions", 1, "arm_id", "C", "arm changed"),
    ("decisions", 0, "decision_at", T0 - timedelta(seconds=1), "decision_at"),
    ("decisions", 0, "send_requested", True, "selection/request"),
    ("decisions", 1, "candidate", False, "selection/request"),
    ("decisions", 1, "selected", False, "selection/request"),
    ("decisions", 1, "expected_cost", "-1", "expected_cost"),
    ("decisions", 1, "expected_cost", "10.000000001", "exceed budget"),
    ("decisions", 1, "expected_cost", None, "lacks expected_cost"),
    ("decisions", 1, "planned_sent_at", END, "planned_sent_at"),
    ("executions", 0, "actual_sent", True, "without request"),
    ("executions", 0, "send_cost_minor", 1, "unsent"),
    ("executions", 1, "failure_reason", None, "failure_reason"),
    ("executions", 3, "sent_at", END, "sent_at"),
    ("executions", 3, "sent_at", T0, "sent_at"),
    ("executions", 3, "discount_rate", ".2", "coupon rate"),
    ("executions", 3, "expires_at", END, "coupon rate or validity"),
    ("executions", 4, "coupon_id", "c-004", "duplicate experimental coupon_id"),
    ("executions", 3, "send_cost_minor", -1, "send_cost_minor"),
    ("outcomes", 1, "outcome_start_at", T0 + timedelta(seconds=1), "outcome window"),
    ("outcomes", 1, "outcome_end_at", END + timedelta(seconds=1), "outcome window"),
    ("outcomes", 2, "censoring_reason", None, "censoring_reason"),
    ("outcomes", 2, "paid_minor", 0, "censored final outcomes"),
    ("outcomes", 2, "purchase_occurred", False, "censored final outcomes"),
    ("outcomes", 0, "paid_minor", 1, "money recorded without purchase"),
    ("outcomes", 3, "send_cost_minor", 3, "disagrees with execution"),
    ("outcomes", 4, "currency", "USD", "currency"),
    ("outcomes", 4, "amount_unit", ".01", "amount_unit"),
    ("outcomes", 4, "raw_database_sha256", SHA_PRE, "raw_database_sha256"),
    ("outcomes", 4, "evaluation_dataset_id", "other", "evaluation_dataset_id"),
])
def test_ledger_corruption_is_rejected(run, ledger, row, field, value, error):
    run[2][ledger].at[row, field] = value
    with pytest.raises(InputValidationError, match=error):
        validate(run)


def test_a_no_action_even_if_marked_candidate(run):
    run[2]["decisions"].loc[0, ["candidate", "selected"]] = True
    with pytest.raises(InputValidationError, match="A cannot select"):
        validate(run)


def test_requested_vs_actual_consistency_for_active_arm(run):
    row = run[2]["decisions"]
    row.loc[3, ["send_requested", "planned_sent_at"]] = [False, None]
    with pytest.raises(InputValidationError, match="actual send without request"):
        validate(run)


@pytest.mark.parametrize("change", ["duplicate column", "unexpected column", "missing column",
                                     "duplicate assignment", "missing outcome", "ineligible assignment"])
def test_schema_and_exact_key_sets(run, change):
    tables = run[2]
    assignment = tables["assignments"]
    if change == "duplicate column":
        tables["assignments"] = pd.concat([assignment, assignment[["arm_id"]]], axis=1)
    elif change == "unexpected column":
        tables["assignments"] = assignment.assign(posthoc_response_type=1)
    elif change == "missing column":
        tables["assignments"] = assignment.drop(columns="arm_id")
    elif change == "duplicate assignment":
        duplicate = assignment.iloc[[0]].assign(snapshot_id="other-snapshot")
        tables["assignments"] = pd.concat([assignment, duplicate])
    elif change == "missing outcome":
        tables["outcomes"] = tables["outcomes"].iloc[:-1]
    else:
        assignment.loc[0, ["customer_id", "snapshot_id"]] = ["excluded", "s-excluded"]
    with pytest.raises(InputValidationError):
        validate(run)


def test_independent_assignment_allows_realized_imbalance(run):
    _, manifest, tables = run
    for name in ("assignments", "decisions", "executions", "outcomes"):
        tables[name].at[2, "arm_id"] = "B"
    manifest["arms"]["B"]["budget"] = 20
    manifest["arms"]["C"]["budget"] = 0
    result = validate(run)
    assert result.by_arm["B"]["itt"] == 2 and result.by_arm["C"]["itt"] == 0
    manifest["assignment_method"] = "fixed_largest_remainder"
    with pytest.raises(InputValidationError, match="fixed quota"):
        validate(run)


def test_fixed_quota_marginals_and_rounding(run):
    spec, manifest, tables = run
    spec = replace(spec, allocation_weights=(2, 1, 1, 1, 1))
    manifest["assignment_method"] = "fixed_largest_remainder"
    # Five places and weights 2:1:1:1:1 round to one place in each arm.
    assert validate((spec, manifest, tables)).by_arm["A"]["itt"] == 1
    tables["assignments"].at[0, "assignment_probability"] = 2 / 6
    with pytest.raises(InputValidationError, match="assignment_probability"):
        validate((spec, manifest, tables))


def test_unobserved_when_coverage_ends_before_window(run):
    run[1]["observation_end_at"] = END - timedelta(seconds=1)
    with pytest.raises(InputValidationError, match="invalid completed observation"):
        validate(run)


def test_assignment_anchored_window(run):
    _, manifest, tables = run
    assigned = T0 + timedelta(minutes=5)
    manifest.update(outcome_start_rule="assigned_at", observation_end_at=END + timedelta(minutes=5))
    tables["assignments"]["assigned_at"] = assigned
    tables["decisions"]["decision_at"] = assigned
    tables["outcomes"]["outcome_start_at"] = assigned
    tables["outcomes"]["outcome_end_at"] = assigned + timedelta(days=28)
    assert validate(run).by_arm["D2"]["observed"] == 1
    tables["assignments"].at[4, "assigned_at"] = assigned + timedelta(minutes=1)
    with pytest.raises(InputValidationError, match="decision_at"):
        validate(run)


def test_as_of_outcomes_cannot_include_pre_assignment_time(run):
    run[2]["assignments"]["assigned_at"] = T0 + timedelta(seconds=1)
    with pytest.raises(InputValidationError, match="simultaneous assignment"):
        validate(run)


@pytest.mark.parametrize("start_rule", ["as_of_at", "assigned_at"])
def test_static_decision_at_outcome_start_passes_but_day_seven_fails(run, start_rule):
    _, manifest, tables = run
    start = T0 if start_rule == "as_of_at" else T0 + timedelta(minutes=5)
    end = start + timedelta(days=28)
    manifest.update(outcome_start_rule=start_rule, observation_end_at=end)
    tables["assignments"]["assigned_at"] = start
    tables["decisions"]["decision_at"] = start
    tables["outcomes"]["outcome_start_at"] = start
    tables["outcomes"]["outcome_end_at"] = end
    assert validate(run).by_arm["D2"]["sent"] == 1

    late = start + timedelta(days=7)
    tables["decisions"].loc[4, ["decision_at", "planned_sent_at"]] = [late, late]
    tables["executions"].loc[4, ["sent_at", "expires_at"]] = [late, late + timedelta(days=28)]
    with pytest.raises(InputValidationError, match="decision_at.*outcome_start_at"):
        validate(run)


def test_static_decision_does_not_override_existing_send_window(run):
    executions = run[2]["executions"]
    assert validate(run).by_arm["D2"]["sent"] == 1  # Scheduled after observation begins.
    executions.loc[4, ["sent_at", "expires_at"]] = [END, END + timedelta(days=28)]
    with pytest.raises(InputValidationError, match="invalid sent_at"):
        validate(run)


@pytest.mark.parametrize("field", ["strategy_version", "model_version", "cost_model_version",
                                  "score_definition_id", "solver_kind"])
def test_d1_d2_comparison_contract(run, field):
    run[1]["arms"]["D2"][field] = "different"
    with pytest.raises(InputValidationError, match="D1/D2"):
        validate(run)


def test_power_plan_shortfall_is_readiness_warning(run):
    spec = run[0]
    plan = plan_sample_size(spec, tuple(PowerScenario((a, "D2"), .1, .02, "increase", .1, "fixture")
                                       for a in ("B", "C")))
    result = validate(run, power_plan=plan)
    assert len(result.readiness_warnings) == 5
    assert result.by_arm["C"]["sample_plan"] == {
        "assigned_target": 5160, "observed_target": 4644,
        "assigned_shortfall": 5159, "observed_shortfall": 4644,
    }
    assert result.data_integrity == "passed"
    with pytest.raises(InputValidationError, match="power_plan context"):
        validate(run, power_plan=replace(plan, version="other"))
    counts = tuple((arm, n + (arm == "A")) for arm, n in plan.required_assigned_by_arm)
    with pytest.raises(InputValidationError, match="allocation weights"):
        validate(run, power_plan=replace(plan, required_assigned_by_arm=counts))


def test_provided_evidence_does_not_confer_approval(run):
    policy = {"B": {"artifact_id": "B-artifact", "selected_keys": [("NA", "s-NA")]}}
    result = validate(run, training_customer_ids={"train-only"}, policy_evidence=policy)
    assert result.evidence_status["training_overlap"] == "no_overlap_in_supplied_ids"
    assert result.evidence_status["policy_decisions"] == "partial_arm_selections_match_supplied_evidence"
    assert result.evidence_status["strategy_approval"] == "unverified"
    assert result.evidence_status["randomization"] == "unverified"
    assert result.evidence_status["producer_file_integrity"] == "unverified"
    with pytest.raises(InputValidationError, match="overlap"):
        validate(run, training_customer_ids={"excluded"})
    policy["B"]["selected_keys"] = [("001", "s-001")]
    with pytest.raises(InputValidationError, match="decision artifact or selections"):
        validate(run, policy_evidence=policy)


def feature_frame(run):
    roster = run[2]["population"]
    frame = roster[["evaluation_dataset_id", "customer_id", "snapshot_id", "as_of_at",
                    "raw_database_sha256", "currency", "amount_unit"]].rename(
                        columns={"evaluation_dataset_id": "dataset_id"})
    return frame.assign(reference_week="2026-01-05", source_model_sha256="d" * 64,
                        feature_set_version="kit32-v1", behavior_feature_set_version=BEHAVIOR_FEATURE_VERSION,
                        expanded_feature_set_version=EXPANDED_FEATURE_VERSION,
                        **dict.fromkeys(EXPANDED_FEATURES, None))


def test_feature_keys_and_input_immutability(run):
    _, manifest, tables = run
    manifest_before = deepcopy(manifest)
    tables["outcomes"] = tables["outcomes"].iloc[::-1].copy()
    tables["outcomes"].attrs["fixture"] = {"labels": ["synthetic"]}
    before = deepcopy(tables)
    features = feature_frame(run).iloc[::-1].copy()
    features_before = features.copy(deep=True)
    result = validate(run, features=features)
    assert result.evidence_status["feature_context"] == "supplied_feature_context_matches"
    assert manifest == manifest_before
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, before[name])
        assert frame.attrs == before[name].attrs
    pd.testing.assert_frame_equal(features, features_before)


@pytest.mark.parametrize("field,value,error", [
    ("as_of_at", T0 + timedelta(hours=1), "features.as_of_at"),
    ("snapshot_id", "unsupplied", "evaluation roster"),
    ("raw_database_sha256", SHA_POST, "raw_database_sha256"),
    ("purchase_occurred", True, "unexpected columns"),
])
def test_features_reject_future_columns_and_provenance_mismatch(run, field, value, error):
    features = feature_frame(run)
    features[field] = value
    with pytest.raises(InputValidationError, match=error):
        validate(run, features=features)
