"""Synthetic purchase experiment. Its known signal is only a software test."""
from dataclasses import replace
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest

from final_strategy.policy_model import ResidualConfig, SplitEvidence, analyze_uplift
from final_strategy.policy_value import ASSIGNMENT_DESIGN, INTERVENTION_SPEC, ExperimentEvidence
from final_strategy.tests.test_segmentation import make_inputs, T0


@pytest.fixture(scope="module")
def case():
    records, roles, rows = [], [], []
    for role in ("train", "discovery", "confirmation", "selection", "final"):
        for j in range(40):
            uid = f"{role}-{j:02}"
            a = j % 2
            # High uplift group buys under Treatment, low group under Control.
            upper = j % 4 >= 2
            purchase = bool(a == upper)
            records.append((uid, .8 if j % 8 >= 4 else .2, .3 if upper else .1, j+1))
            roles.append((uid, role))
            rows.append(dict(customer_id=uid, dataset_id="dummy-segmentation", snapshot_id=f"opaque:{uid}:{T0.isoformat()}",
                             as_of_at=T0, assigned_at=T0, currency="SOURCE_UNSPECIFIED", amount_unit=".01",
                             experiment_id="fixture", treatment=a, assignment_probability=.5, coupon_action=a,
                             actual_sent=bool(a), coupon_id=uid if a else None, execution_status="sent" if a else "not_requested",
                             sent_at=T0 if a else None, expires_at=T0+pd.Timedelta(days=28) if a else None,
                             discount_rate=.1 if a else None, additional_send_count=0, eligible_at_assignment=True,
                             outcome_end_at=T0+pd.Timedelta(days=28), outcome_observed=True, is_censored=False,
                             factual_no_order=not purchase, factual_no_session=not purchase,
                             factual_outcome_net_revenue_minor=int(purchase), send_cost_minor=a,
                             outcome_redemption_discount_cost_minor=0))
    opts = dict(evidence=ExperimentEvidence("dummy-segmentation", "fixture", "a"*64, "synthetic",
                                           ASSIGNMENT_DESIGN, INTERVENTION_SPEC),
                split=SplitEvidence("synthetic", "generated fixture", 43, "synthetic upstream", ()),
                roles=pd.DataFrame(roles, columns=["customer_id", "role"]), min_arm=2, min_events=0,
                cuts=(.1,), quantiles=(.5,), allow_unverified_log=True)
    return make_inputs(records), pd.DataFrame(rows), opts


def analyze(case, **opts):
    inputs, log, defaults = case
    return analyze_uplift(inputs, log, **(defaults | opts))


def test_hand_calibration_and_boundaries(case):
    inputs, log, _ = case
    before, source = inputs.data.copy(deep=True), log.copy(deep=True)
    result = analyze(case)
    bins = [b for b in result["bins"]["confirmation"] if b["churn"] is None and b["clv"] is None]
    assert [b["n"] for b in bins] == [20, 0, 20]
    assert [b["observed_difference"] for b in bins] == [-1, None, 1]
    assert bins[0]["control_events"] == 10 and bins[0]["treatment_events"] == 0
    assert bins[0]["mean_residual"] == pytest.approx(-1.1)
    assert not bins[1]["supported"]
    # Four equally frequent (A,Y) combinations: Z = -2,0,0,+2.
    assert result["diagnostics"]["confirmation"]["baseline_mse"] == pytest.approx(
        ((-2-.1)**2 + (.1)**2 + (.3)**2 + (2-.3)**2) / 4)
    assert result["split"]["counts"]["final"] == 40
    json.dumps(result, allow_nan=False)
    pd.testing.assert_frame_equal(inputs.data, before)
    pd.testing.assert_frame_equal(log, source)


@pytest.mark.parametrize("name", ["treatment", "residual", "factual_no_session", "true_tau"])
def test_predictor_allowlist(case, name):
    with pytest.raises(ValueError, match="KIT39"):
        analyze(case, extra_features=(name,))


@pytest.mark.parametrize("field,value", [("snapshot_id", "wrong"), ("assignment_probability", .4),
                                        ("is_censored", True), ("factual_no_order", 0)])
def test_log_validation(case, field, value):
    inputs, log, opts = case
    changed = log.copy()
    changed[field] = changed[field].astype(object)
    changed.at[0, field] = value
    with pytest.raises(ValueError):
        analyze_uplift(inputs, changed, **opts)


def test_split_coverage_and_upstream_overlap(case):
    _, _, opts = case
    roles = opts["roles"]
    for invalid in (roles.iloc[1:], pd.concat([roles, roles.iloc[:1]])):
        with pytest.raises(ValueError, match="roster"):
            analyze(case, roles=invalid)
    for customer, message in [("final-00", "reserved"), ("train-00", "OOF")]:
        with pytest.raises(ValueError, match=message):
            analyze(case, split=replace(opts["split"], upstream_customer_ids=(customer,)))


def test_optional_pre_features_and_oof_declaration(case):
    _, _, opts = case
    split = replace(opts["split"], upstream_customer_ids=("train-00",), oof_source="synthetic-fold-record")
    result = analyze(case, extra_features=("mean_net_order_value_30d",), split=split)
    assert result["features"] == ["churn_probability", "uplift_score", "predicted_clv_12m", "mean_net_order_value_30d"]
    assert result["split"]["independence"] == "not_independently_verified"
    assert result["split"]["upstream_roster_sha256"] and result["split"]["oof_source"] == split.oof_source


def test_log_roster_and_units(case):
    inputs, log, options = case
    for invalid in (log.iloc[:-2], pd.concat([log.iloc[:-1], log.iloc[:1]]), log.assign(amount_unit="1")):
        with pytest.raises(ValueError):
            analyze_uplift(inputs, invalid, **options)


def test_reserved_outcomes_cannot_change_diagnostics(case):
    inputs, log, opts = case
    changed = log.copy()
    held = changed.customer_id.str.startswith(("selection", "final"))
    changed.loc[held, "factual_no_order"] = True
    changed.loc[held, "factual_outcome_net_revenue_minor"] = 0
    assert analyze_uplift(inputs, changed, **opts) == analyze(case)


def test_censored_cohort_rejected(case):
    inputs, log, opts = case
    changed = log.copy()
    changed.loc[0, ["is_censored", "outcome_observed"]] = [True, False]
    for field in ("factual_no_order", "factual_no_session", "factual_outcome_net_revenue_minor"):
        changed[field] = changed[field].astype(object)
        changed.at[0, field] = None
    with pytest.raises(ValueError, match="complete 28-day"):
        analyze_uplift(inputs, changed, **opts)


def test_missing_xgboost(case, monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ImportError, match="optional xgboost"):
        analyze(case, model_config=ResidualConfig())


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="optional xgboost")
def test_actual_tree_signal_and_reproducibility(case):
    config = ResidualConfig()
    result = analyze(case, model_config=config)
    assert analyze(case, model_config=config) == result
    check = result["diagnostics"]["confirmation"]
    assert check["adjusted_mse"] < check["baseline_mse"]
    assert result["residual_model"]["splits"]
    assert "xgboost_split" in {t["source"] for t in result["thresholds"]}
    assert result["policy_effect_verified"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("wrap", [tuple, list, iter, lambda names: (f for f in names)])
def test_single_pass_allowlist(case, wrap):
    with pytest.raises(ValueError, match="KIT39"):
        analyze(case, extra_features=wrap(["residual"]), model_config=ResidualConfig())


@pytest.mark.parametrize("features", ["", "residual", [None], [["residual"]],
                                      ["mean_net_order_value_30d"] * 2])
def test_rejects_invalid_feature_sequence(case, features):
    with pytest.raises(ValueError, match="extra_features|feature names"):
        analyze(case, extra_features=features)


def test_iterable_configuration_matches_tuple(case):
    opts = dict(extra_features=("mean_net_order_value_30d",), cuts=(.1,), quantiles=(.5,), churn_cuts=(.6,))
    assert analyze(case, **opts) == analyze(case, **{k: iter(v) for k, v in opts.items()})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 3, True, "", "  ", []])
def test_invalid_oof_source_without_overlap(case, bad):
    with pytest.raises(ValueError, match="OOF source"):
        analyze(case, split=replace(case[2]["split"], oof_source=bad))
