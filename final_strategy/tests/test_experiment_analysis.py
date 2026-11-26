"""Independent numerical checks on synthetic ledgers, without experiment claims."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from decimal import Decimal
import json
import math
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chi2_contingency

from final_strategy import experiment_analysis as analysis
from final_strategy.experiment_power import PowerScenario, plan_sample_size
from final_strategy.experiment_spec import ExperimentSpec
from final_strategy.experiment_validation import CONTRACT_VERSION
from final_strategy.validation import InputValidationError


T0 = pd.Timestamp("2026-01-04T15:00:00Z")
END = T0 + timedelta(days=28)
ARMS = ("A", "B", "C", "D1", "D2")
SIZES = dict(zip(ARMS, (48, 50, 60, 56, 60)))
PURCHASES = dict(zip(ARMS, (15, 15, 26, 30, 38)))
CHURN = dict(zip(ARMS, (24, 22, 26, 20, 12)))
PRE, POST = "a" * 64, "b" * 64


@pytest.fixture
def experiment():
    spec = ExperimentSpec("synthetic", "v1", (1,) * 5, family_alpha=np.float32(.05))
    ledger_names = ("population", "assignments", "decisions", "executions", "outcomes")
    manifest = {
        "contract_version": CONTRACT_VERSION, "experiment_id": spec.experiment_id, "version": spec.version,
        "evaluation_dataset_id": "synthetic-eval", "as_of_at": T0, "population_frozen_at": T0,
        "observation_start_at": T0 - timedelta(days=28), "observation_end_at": END,
        "outcome_start_rule": "as_of_at", "currency": "KRW", "amount_unit": Decimal("1"),
        "eligibility_policy_version": "elig-v1", "coupon_policy_version": "coupon-v1",
        "assignment_algorithm": "declared-fixture", "assignment_seed": np.int64(42),
        "assignment_version": "v1", "assignment_method": "independent",
        "feature_raw_sha256": PRE, "result_raw_sha256": POST,
        "ledger_sha256": {name: "c" * 64 for name in ledger_names}, "budget_per_customer": "10",
        "arms": {arm: {
            "strategy_version": "full-v1" if arm.startswith("D") else arm,
            "model_version": "m1", "score_definition_id": "s1", "cost_model_version": "c1",
            "solver_kind": {"D1": "greedy", "D2": "milp"}.get(arm, "none"), "solver_version": "v1",
            "decision_artifact_id": arm + "-artifact", "budget": 0 if arm == "A" else 10 * SIZES[arm],
        } for arm in ARMS},
    }
    rows = {name: [] for name in ledger_names}
    for arm in ARMS:
        for i in range(SIZES[arm]):
            uid = f"{arm}-{i:03d}"
            context = dict(evaluation_dataset_id="synthetic-eval", customer_id=uid, snapshot_id="s-" + uid,
                           as_of_at=T0, raw_database_sha256=PRE, currency="KRW", amount_unit=1)
            rows["population"].append(dict(context, eligible=True, eligibility_at=T0, exclusion_reason=None))
            context["arm_id"] = arm
            rows["assignments"].append(dict(context, assigned_at=T0, assignment_probability=np.float32(.2)))
            requested = arm != "A" and i < 2
            sent = requested and i == 0
            rows["decisions"].append(dict(
                context, decision_at=T0, candidate=requested, selected=requested, send_requested=requested,
                planned_sent_at=T0 if requested else None, expected_cost="2.000000001" if requested else None,
            ))
            context["raw_database_sha256"] = POST
            rows["executions"].append(dict(
                context, actual_sent=sent, sent_at=T0 if sent else None, discount_rate=".1" if sent else None,
                coupon_id="c-" + uid if sent else None, expires_at=END if sent else None,
                execution_status="sent" if sent else "failed" if requested else "not_requested",
                failure_reason="synthetic failure" if requested and not sent else None,
                send_cost_minor=np.int64(4 if sent else 0),
            ))
            purchase = i < PURCHASES[arm]
            rows["outcomes"].append(dict(
                context, outcome_start_at=T0, outcome_end_at=END, outcome_observed=np.bool_(True),
                censoring_reason=None, purchase_occurred=np.bool_(purchase),
                session_occurred=np.bool_(i >= CHURN[arm]), paid_minor=np.int64(25 if purchase else 0),
                coupon_discount_minor=np.int64(3 if purchase and sent else 0),
                send_cost_minor=np.int64(4 if sent else 0),
            ))
    return spec, manifest, {name: pd.DataFrame(items, dtype=object) for name, items in rows.items()}


def analyze(experiment, **kwargs):
    spec, manifest, ledgers = experiment
    return analysis.analyze_experiment(spec, manifest, **ledgers, **kwargs)


def reference_ci(x1, n1, x2, n2):
    # SciPy's independent single-proportion Wilson implementation, then
    # Newcombe's root-sum-of-squares combination of the one-sided deviations.
    w1 = binomtest(x1, n1).proportion_ci(.95, method="wilson")
    w2 = binomtest(x2, n2).proportion_ci(.95, method="wilson")
    p, q = x1 / n1, x2 / n2
    return (p - q - math.sqrt((p - w1.low)**2 + (w2.high - q)**2),
            p - q + math.sqrt((w1.high - p)**2 + (q - w2.low)**2))


def test_full_itt_and_independent_statistical_oracles(experiment):
    result = analyze(experiment)
    b, c = result.primary_comparisons
    assert [(r["baseline_arm"], r["comparison_arm"]) for r in (b, c)] == [("B", "D2"), ("C", "D2")]
    assert (b["baseline_events_observed"], b["comparison_events_observed"]) == (15, 38)
    assert (b["baseline_n_itt"], b["comparison_n_itt"]) == (50, 60)
    difference = 38 / 60 - 15 / 50
    pooled = 53 / 110
    reference_z = difference / math.sqrt(pooled * (1 - pooled) * (1 / 50 + 1 / 60))
    pearson = chi2_contingency([[38, 22], [15, 35]], correction=False)
    assert b["absolute_difference"] == pytest.approx(difference)
    assert b["absolute_difference_display_unit"] == "%p"
    assert b["relative_change"] == pytest.approx(difference / .3)
    assert b["z_statistic"] == pytest.approx(reference_z)
    assert b["z_statistic"]**2 == pytest.approx(pearson.statistic)
    assert b["raw_p_value"] == pytest.approx(math.erfc(reference_z / math.sqrt(2)))
    assert b["raw_p_value"] == pytest.approx(pearson.pvalue)
    assert b["confidence_interval"] == pytest.approx(reference_ci(38, 60, 15, 50))
    assert b["multiplicity_adjusted"] is False and b["confidence_level"] == .95
    pearson_c = chi2_contingency([[38, 22], [26, 34]], correction=False)
    assert c["raw_p_value"] == pytest.approx(pearson_c.pvalue)
    assert c["confidence_interval"] == pytest.approx(reference_ci(38, 60, 26, 60))
    assert b["adjusted_p_value"] == pytest.approx(2 * pearson.pvalue)
    assert c["adjusted_p_value"] == pytest.approx(max(2 * pearson.pvalue, pearson_c.pvalue))
    assert all(0 <= row[k] <= 1 for row in (b, c) for k in ("raw_p_value", "adjusted_p_value"))
    assert b["confirmatory_decision"] == c["confirmatory_decision"] == "reject"
    assert result.holm["status"] == "applied"
    d1, = result.secondary_comparisons
    assert d1["baseline_arm"] == "D1" and d1["scope"] == "exploratory"
    assert "adjusted_p_value" not in d1 and "confirmatory_decision" not in d1
    churn = result.exploratory_churn_comparisons[0]
    assert (churn["baseline_events_observed"], churn["comparison_events_observed"]) == (22, 12)
    assert churn["absolute_difference"] == pytest.approx(.44 - .20)
    assert churn["relative_change"] == pytest.approx((.44 - .20) / .44)
    assert churn["z_statistic"] > 0
    assert churn["confidence_interval"] == pytest.approx(reference_ci(22, 50, 12, 60))
    assert "confirmatory_decision" not in churn and churn["scope"] == "exploratory"
    assert all(r["baseline_arm"] != "A" for r in result.exploratory_churn_comparisons)
    assert (result.by_arm["B"]["n_itt"], result.by_arm["B"]["n_sent"], result.by_arm["B"]["n_failed"]) == (50, 1, 1)


def _set_purchases(rows, mask, value):
    rows.loc[mask, "purchase_occurred"] = value
    rows.loc[mask, "paid_minor"] = 25 if value else 0
    if not value:
        rows.loc[mask, "coupon_discount_minor"] = 0


def test_negative_purchase_and_churn_effects_keep_their_directions(experiment):
    rows = experiment[2]["outcomes"]
    _set_purchases(rows, rows.arm_id.eq("B"), True)
    rows.loc[rows.arm_id.eq("B"), "session_occurred"] = True
    rows.loc[rows.customer_id.eq("B-000"), "session_occurred"] = False
    result = analyze(experiment)
    for row, expected in ((result.primary_comparisons[0], reference_ci(38, 60, 50, 50)),
                          (result.exploratory_churn_comparisons[0], reference_ci(1, 50, 12, 60))):
        assert row["absolute_difference"] < 0 and row["z_statistic"] < 0
        assert row["confidence_interval"] == pytest.approx(expected)
        assert row["confidence_interval"][1] < 0


@pytest.mark.parametrize("purchased", [False, True])
def test_all_outcomes_at_boundary_have_ci_but_no_p_value(experiment, purchased):
    rows = experiment[2]["outcomes"]
    _set_purchases(rows, rows.index, purchased)
    b = analyze(experiment).primary_comparisons[0]
    assert b["absolute_difference"] == 0
    assert b["confidence_interval"] == pytest.approx(reference_ci(60 if purchased else 0, 60,
                                                                 50 if purchased else 0, 50))
    assert b["reason"] == "pooled_rate_at_boundary"
    assert b["raw_p_value"] is None and b["adjusted_p_value"] is None
    assert b["confirmatory_decision"] == "withheld"


def test_zero_baseline_is_valid_when_expected_counts_support_z_test(experiment):
    rows = experiment[2]["outcomes"]
    _set_purchases(rows, rows.arm_id.eq("B"), False)
    comparison = analyze(experiment).primary_comparisons[0]
    assert comparison["baseline_rate_itt"] == 0
    assert comparison["relative_change"] is None
    assert comparison["test_status"] == "tested" and comparison["raw_p_value"] > 0


def test_sparse_expected_cells_keep_effect_and_hold_family(experiment):
    rows = experiment[2]["outcomes"]
    _set_purchases(rows, rows.arm_id.isin(("B", "D2")), False)
    _set_purchases(rows, rows.customer_id.eq("D2-002"), True)
    result = analyze(experiment)
    b, c = result.primary_comparisons
    assert b["reason"] == "expected_cell_below_five"
    assert b["absolute_difference"] == pytest.approx(1 / 60)
    assert b["confidence_interval"] is not None and b["raw_p_value"] is None
    assert c["raw_p_value"] is not None
    assert b["adjusted_p_value"] is None and c["adjusted_p_value"] is None
    assert result.holm["unavailable_comparisons"][0]["baseline_arm"] == "B"


@pytest.mark.parametrize("keep", [0, 2])
def test_empty_or_very_small_arm(experiment, keep):
    _, manifest, ledgers = experiment
    excluded = {f"B-{i:03d}" for i in range(keep, SIZES["B"])}
    for name, rows in ledgers.items():
        ledgers[name] = rows.loc[~rows.customer_id.isin(excluded)].copy()
    manifest["arms"]["B"]["budget"] = 10 * keep
    b = analyze(experiment).primary_comparisons[0]
    assert b["baseline_n_itt"] == keep
    assert b["reason"] == ("expected_cell_below_five" if keep else "empty_arm")
    assert b["raw_p_value"] is None and b["confirmatory_decision"] == "withheld"
    assert (b["confidence_interval"] is None) == (keep == 0)


def test_equal_rates_and_holm_ties(experiment):
    rows = experiment[2]["outcomes"]
    # Both B and C have the same 0.6 outcome rate as D2.
    _set_purchases(rows, rows.arm_id.isin(("B", "C", "D2")), False)
    for arm in ("B", "C", "D2"):
        keys = {f"{arm}-{i:03d}" for i in range(3 * SIZES[arm] // 5)}
        _set_purchases(rows, rows.customer_id.isin(keys), True)
    result = analyze(experiment)
    assert [r["baseline_arm"] for r in result.primary_comparisons] == ["B", "C"]
    for row in result.primary_comparisons:
        assert row["z_statistic"] == 0
        assert row["raw_p_value"] == row["adjusted_p_value"] == 1
        assert row["confirmatory_decision"] == "do_not_reject"


@pytest.mark.parametrize("arm", ["B", "C", "D2"])
def test_censoring_preserves_itt_and_holds_holm(experiment, arm):
    rows = experiment[2]["outcomes"]
    target = rows.customer_id.eq(arm + "-000")  # An observed purchaser who was sent a coupon.
    rows.loc[target, ["outcome_observed", "censoring_reason", "purchase_occurred", "session_occurred",
                      "paid_minor", "coupon_discount_minor"]] = [False, "synthetic gap", None, None, None, None]
    result = analyze(experiment)
    summary = result.by_arm[arm]
    assert summary["n_itt"] == SIZES[arm]
    assert (summary["n_observed"], summary["n_censored"]) == (SIZES[arm] - 1, 1)
    assert summary["purchase_rate_itt"] is None and summary["churn_rate_itt"] is None
    assert summary["observed_only_descriptive"]["purchase_rate"] == pytest.approx(
        (PURCHASES[arm] - 1) / (SIZES[arm] - 1))
    assert result.holm["status"] == "withheld"
    assert summary["money"]["observed_paid_minor"] == 25 * (PURCHASES[arm] - 1)
    assert summary["money"]["itt_paid_minor"] is None
    assert summary["money"]["send_cost_minor"] == 4
    assert summary["money"]["realized_spend_minor"] is None
    assert summary["money"]["realized_spend_minor_per_customer"] is None
    for row in result.primary_comparisons:
        assert row["adjusted_p_value"] is None and row["confirmatory_decision"] == "withheld"
        if arm in (row["baseline_arm"], row["comparison_arm"]):
            assert row["absolute_difference"] is row["confidence_interval"] is row["raw_p_value"] is None
            assert row["reason"] == "incomplete_itt_observation"
        else:
            assert row["raw_p_value"] is not None
    assert summary["observation_rate"] < result.by_arm["A"]["observation_rate"]


def test_money_exact_amounts_and_realized_overage(experiment):
    _, manifest, ledgers = experiment
    # Raise an observed send fee without altering the declared plan.
    for name in ("executions", "outcomes"):
        rows = ledgers[name]
        rows.loc[rows.customer_id.eq("D2-000"), "send_cost_minor"] = 601
    result = analyze(experiment)
    money = result.by_arm["D2"]["money"]
    assert result.currency == manifest["currency"]
    assert money["itt_paid_minor"] == 950  # Already net of discounts.
    assert money["observed_paid_minor"] == 950
    assert money["itt_denominator"] == money["observed_denominator"] == 60
    assert Decimal(money["itt_paid_minor_per_customer"]) == Decimal("15.83333333333333333333333333333333")
    assert money["send_cost_minor"] == 601 and money["itt_coupon_discount_minor"] == 3
    assert money["realized_spend_minor"] == 604
    assert money["expected_cost_minor"] == "4.000000002"
    assert money["realized_minus_expected_minor"] == "599.999999998"
    assert money["budget_minor"] == "600" and money["realized_budget_status"] == "over_budget"
    out = ledgers["outcomes"]
    out.loc[out.customer_id.eq("D2-001"), ["outcome_observed", "censoring_reason", "purchase_occurred",
             "session_occurred", "paid_minor", "coupon_discount_minor"]] = [False, "gap", None, None, None, None]
    money = analyze(experiment).by_arm["D2"]["money"]
    assert money["known_spend_minor"] == 604 and money["realized_budget_status"] == "over_budget"
    assert money["realized_spend_minor"] is money["realized_minus_expected_minor"] is None


@pytest.mark.parametrize("ledger,field,value", [
    ("outcomes", "snapshot_id", "wrong"),
    ("assignments", "evaluation_dataset_id", "wrong"),
    ("outcomes", "raw_database_sha256", PRE),
    ("outcomes", "currency", "USD"),
    ("outcomes", "amount_unit", Decimal(".01")),
    ("outcomes", "paid_minor", float("nan")),
    ("outcomes", "paid_minor", float("inf")),
])
def test_invalid_ledger_rejected_by_phase2(experiment, ledger, field, value):
    experiment[2][ledger].iat[0, experiment[2][ledger].columns.get_loc(field)] = value
    with pytest.raises(InputValidationError):
        analyze(experiment)


def test_duplicate_customer_rejected(experiment):
    rows = experiment[2]["outcomes"]
    experiment[2]["outcomes"] = pd.concat([rows, rows.iloc[[0]]])
    with pytest.raises(InputValidationError, match="duplicate customer"):
        analyze(experiment)


def test_strict_json_nested_immutability_evidence_and_reordering(experiment):
    spec, manifest, ledgers = experiment
    before = deepcopy(ledgers)
    manifest_before = deepcopy(manifest)
    power = plan_sample_size(spec, tuple(PowerScenario((a, "D2"), .1, .02, "increase", .1, "fixture")
                                        for a in ("B", "C")))
    options = dict(power_plan=power, training_customer_ids={"train-only"},
                   policy_evidence={"B": {"artifact_id": "B-artifact",
                                          "selected_keys": [("B-000", "s-B-000"), ("B-001", "s-B-001")]}})
    result = analyze(experiment, **options)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["validation"]["design_status"] == "planning_unapproved"
    assert payload["validation"]["data_integrity"] == "passed"
    assert payload["validation"]["readiness_warnings"]
    assert payload["validation"]["evidence_status"]["training_overlap"] == "no_overlap_in_supplied_ids"
    assert payload["validation"]["evidence_status"]["policy_decisions"].startswith("partial")
    assert payload["validation"]["evidence_status"]["randomization"] == "unverified"
    assert payload["causal_interpretation"] == "unverified"
    assert all(result.by_arm[a]["n_itt"] == result.by_arm[a]["n_observed"] + result.by_arm[a]["n_censored"]
               for a in ARMS)
    with pytest.raises(TypeError):
        result.by_arm["B"]["money"]["itt_paid_minor"] = 0
    with pytest.raises(FrozenInstanceError):
        result.currency = "USD"
    payload["by_arm"]["B"]["money"]["itt_paid_minor"] = -1
    shuffled = {name: frame.sample(frac=1, random_state=i) for i, (name, frame) in enumerate(ledgers.items())}
    mapped = MappingProxyType(dict(manifest, ledger_sha256=MappingProxyType(manifest["ledger_sha256"])))
    assert result.to_dict() == analyze((spec, mapped, shuffled), **options).to_dict()
    for name, frame in ledgers.items():
        pd.testing.assert_frame_equal(frame, before[name])
    assert manifest == manifest_before


def test_declared_secondary_adds_only_requested_a_comparison(experiment):
    spec, manifest, ledgers = experiment
    spec = replace(spec, secondary_comparisons=(("D1", "D2"), ("A", "B")))
    result = analyze((spec, manifest, ledgers))
    assert [(r["baseline_arm"], r["comparison_arm"]) for r in result.secondary_comparisons] == [
        ("D1", "D2"), ("A", "B")]
    assert [(r["baseline_arm"], r["comparison_arm"]) for r in result.exploratory_churn_comparisons] == [
        ("B", "D2"), ("C", "D2"), ("D1", "D2"), ("A", "B")]
    assert len(result.primary_comparisons) == 2


def test_analyzer_uses_same_copies_after_phase2_returns(experiment, monkeypatch):
    _, manifest, ledgers = experiment
    original = analysis.validate_experiment_run

    def validate_then_mutate_original(spec, internal_manifest, **internal_ledgers):
        assert internal_manifest["arms"] is not manifest["arms"]
        assert internal_ledgers["outcomes"] is not ledgers["outcomes"]
        validated = original(spec, internal_manifest, **internal_ledgers)
        # Changes after validation affect rates/provenance with the same row keys and counts.
        manifest["ledger_sha256"]["outcomes"] = "f" * 64
        ledgers["outcomes"].loc[ledgers["outcomes"].arm_id.eq("D2"), "purchase_occurred"] = False
        return validated

    monkeypatch.setattr(analysis, "validate_experiment_run", validate_then_mutate_original)
    result = analyze(experiment)
    assert result.by_arm["D2"]["purchase_customers_observed"] == 38
    assert result.provenance["ledger_sha256"]["outcomes"] == "c" * 64
