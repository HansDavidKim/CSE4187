"""Synthetic request, actual charge and redeemed-order evidence; no seed43 outcomes."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json

import numpy as np
import pandas as pd
import pytest

from final_strategy import coupon_cost_estimation as estimator
from final_strategy.budget import Budget, optimize_budget
from final_strategy.policy_search import run_policy_search
from final_strategy.retention_strategy import COST_INPUT_COLUMNS
from final_strategy.strategy_profiles import build_strategy_profiles
from final_strategy.tests.test_policy_search import RECORDS, _inputs
from final_strategy.tests.test_retention_strategy import run, setup
from final_strategy.tests.test_segmentation import T0, make_inputs
from final_strategy.tests.test_strategy_profiles import CUTS
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError, PREDICTION_COLUMNS
from final_strategy.what_if import run_what_if


START = T0 - pd.Timedelta(weeks=6)
END = START + pd.Timedelta(weeks=4)


@pytest.fixture
def data():
    target = make_inputs(RECORDS).data.drop(columns=list(PREDICTION_COLUMNS.values()))
    target["purchase_count_30d"] = [0, 4, 4, 0]
    past = target.copy()
    past["customer_id"] = ["a", "b", "c", "d"]
    past["snapshot_id"] = ["s1", "s2", "s3", "s4"]
    past["dataset_id"], past["raw_database_sha256"] = "history", "b" * 64
    past["as_of_at"], past["reference_week"] = START, START.tz_convert("Asia/Seoul").date().isoformat()
    frame = past[list(CONTEXT_COLUMNS)].copy()
    frame["policy_id"], frame["discount_rate"], frame["assigned_at"] = "fixture", ".10", START
    frame["treatment"] = frame["coupon_action"] = [1, 1, 1, 0]
    frame["request_id"] = ["r1", "r2", "r3", None]
    frame["actual_sent"] = [True, True, False, False]
    frame["execution_status"] = ["sent", "sent", "failed", "not_requested"]
    frame["failure_reason"] = [None, None, "delivery", None]
    frame["coupon_id"] = ["c1", "c2", None, None]
    frame["sent_at"], frame["expires_at"] = [START, START, None, None], [END, END, None, None]
    frame["outcome_end_at"], frame["outcome_observed"], frame["is_censored"] = END, True, False
    frame["known_at"] = END
    frame["redeemed_in_outcome"] = [True, False, False, False]
    frame["redemption_order_id"] = ["o1", None, None, None]
    frame["redemption_at"] = [START + pd.Timedelta(days=1), None, None, None]
    frame["charged_minor"] = [2, 2, 1, 0]
    frame["outcome_redemption_discount_cost_minor"] = [3, 0, 0, 0]
    lines = pd.DataFrame([("o1", "a", "c1", frame.redemption_at[0], 67, 1, 2, 25),
                          ("o1", "a", "c1", frame.redemption_at[0], 67, 2, 1, 20)],
                         columns=estimator.ORDER_COLUMNS)
    return dict(exposures=frame, order_lines=lines, historical_features=past, target_features=target,
                cohort=estimator.HistoricalCohort("history", "b"*64, "c"*64, "fixture", "explicit_charge",
                                                  True, 4, START, END),
                config=estimator.EstimationConfig("fixture-v1", 1, 1, 2, 3, 2, 5), frozen_at=T0)


def estimate(data, **config):
    return estimator.estimate_coupon_costs(**(data | {"config": replace(data["config"], **config)}))


def test_request_means_and_copy_isolation(data):
    result = estimate(data)
    stats = result.audit["population"]
    assert (stats["requests"], stats["sent"], stats["used"], stats["charges"], stats["discounts"]) == (3,2,1,5,3)
    assert stats["exact"] == dict(send_cost="5/3", redemption_probability="1/3", expected_discount_if_redeemed="3")
    assert Fraction("1/3") * 3 == Fraction(stats["discounts"], stats["requests"])
    detail = result.audit["applied"]["population"]["expected_cost"]
    assert detail["exact"] == "8/3" and Decimal(detail["output"]) == Decimal("2.672")
    assert Fraction(detail["error"]) == Fraction("2.672") - Fraction(8,3)
    assert tuple(result.cost_inputs) == COST_INPUT_COLUMNS
    result.audit["population"]["charges"] = 99
    result.cost_inputs.loc[0, "send_cost"] = 99
    data["target_features"].loc[0, "customer_id"] = "changed"
    assert result.cost_inputs.iloc[0].send_cost == Decimal("1.67") and result.cost_inputs.iloc[0].customer_id == "001"
    assert json.loads(json.dumps(result.audit, allow_nan=False))["population"]["charges"] == 5


def test_groups_component_fallback_and_insufficient_population(data):
    result = estimate(data, purchase_count_30d_threshold=2)
    table = result.cost_inputs.set_index("customer_id")
    assert table.loc["001", "redemption_probability"] == 1 and table.loc["002", "redemption_probability"] == 0
    high = result.audit["applied"]["at_or_above"]
    assert high["expected_discount_if_redeemed"]["level"] == "population" and high["expected_cost"]["mixed_levels"]
    fallback = estimate(data, purchase_count_30d_threshold=2, min_requests=2)
    assert fallback.audit["applied"]["below"]["send_cost"]["level"] == "population"
    for config in ({"min_requests":4}, {"min_redemptions":2}):
        held = estimate(data, **config)
        assert held.status == "withheld" and len(held.audit["customers"]) == 4
        with pytest.raises(InputValidationError, match="withheld"):
            _ = held.cost_inputs


def test_zero_events_not_zero_conditional_amount(data):
    frame = data["exposures"]
    frame.loc[0, ["redeemed_in_outcome","redemption_order_id","redemption_at",
                  "outcome_redemption_discount_cost_minor"]] = [False,None,pd.NaT,0]
    data["order_lines"] = data["order_lines"].iloc[:0]
    held = estimate(data)
    assert held.audit["population"]["exact"]["redemption_probability"] == "0"
    assert held.audit["population"]["exact"]["expected_discount_if_redeemed"] is None
    assert held.status == "withheld"


@pytest.mark.parametrize("column,value", [
    ("assigned_at", T0), ("assigned_at", START-pd.Timedelta(days=1)), ("known_at", T0+pd.Timedelta(days=1)),
    ("outcome_observed", False), ("is_censored", True), ("outcome_end_at", START),
    ("expires_at", END+pd.Timedelta(days=1)), ("redemption_at", END), ("actual_sent", False),
    ("outcome_redemption_discount_cost_minor", 2), ("charged_minor", True), ("charged_minor", float("nan")),
    ("charged_minor", float("inf")), ("charged_minor", -1), ("charged_minor", ".1"), ("coupon_action", 0),
    ("request_id", "r2"), ("coupon_id", "c2"), ("redemption_order_id", "unknown"), ("discount_rate", ".2"),
    ("raw_database_sha256", "c"*64), ("currency", "KRW"), ("amount_unit", "1"), ("snapshot_id", "bad"),
])
def test_invalid_or_open_cohort_rejected(data, column, value):
    frame = data["exposures"]
    frame[column] = frame[column].astype(object)
    frame.loc[0,column] = value
    with pytest.raises(InputValidationError):
        estimate(data)


@pytest.mark.parametrize("column,value", [("coupon_id","c2"),("customer_id","b"),("paid_minor",64),
                                           ("line_no",2),("purchased_at",END)])
def test_invalid_receipt(data,column,value):
    data["order_lines"].loc[0,column] = value
    with pytest.raises(InputValidationError):
        estimate(data)


@pytest.mark.parametrize("patch", [{"min_requests":True},{"min_redemptions":0},{"discount_places":-1},
                                    {"decimal_places":10},{"decimal_places":4},{"probability_places":float("nan")}])
def test_bad_config(data,patch):
    with pytest.raises(InputValidationError):
        estimate(data, **patch)


def test_overlap_duplicate_and_failure_billing(data):
    with pytest.raises(InputValidationError, match="billing"):
        estimate(data | {"cohort":replace(data["cohort"], failure_charge_allowed=False)})
    data["target_features"].loc[0,"customer_id"] = "a"
    with pytest.raises(InputValidationError, match="overlap"):
        estimate(data)
    data["target_features"].loc[0,"customer_id"] = "002"
    with pytest.raises(InputValidationError, match="duplicate"):
        estimate(data)


def test_repeat_customer_different_request_snapshots(data):
    delta = pd.Timedelta(days=7)
    for name in ("exposures","historical_features"):
        f = data[name]
        f.loc[2,["customer_id","reference_week","as_of_at"]] = ["a",(START+delta).tz_convert("Asia/Seoul").date().isoformat(), START+delta]
    for key in ("assigned_at","outcome_end_at","known_at"):
        data["exposures"].loc[2,key] += delta
    data["cohort"] = replace(data["cohort"], observation_end_at=END+delta)
    assert estimate(data).audit["population"]["requests"] == 3


def test_existing_consumers_share_target_costs(data):
    result = estimate(data, purchase_count_30d_threshold=2)
    options = _inputs()
    options.update(cost_inputs=result.cost_inputs, cost_metadata=result.cost_metadata,
                   budget=Budget(10,"SOURCE_UNSPECIFIED",".01"))
    runs = run_policy_search(candidates=build_strategy_profiles(**CUTS).policies, **options).runs
    assert set(runs) == {"P0","H0","H1","H2","H3","H4-R","H4-V"}
    table = result.cost_inputs.set_index("customer_id")
    expected = table.send_cost + table.redemption_probability * table.expected_discount_if_redeemed
    for plan in runs.values():
        active = plan.decisions.query("budget_candidate").set_index("customer_id")
        assert active.expected_cost.to_dict() == expected.loc[active.index].to_dict()
    bundle = setup(RECORDS)
    keys = bundle[4].customer_id
    bundle[4:6] = [result.cost_inputs.loc[result.cost_inputs.customer_id.isin(keys)], result.cost_metadata]
    retention = run(bundle)
    args = (retention.budget_inputs,retention.costs,options["budget"],result.cost_metadata)
    kwargs = dict(dataset_id=options["dataset_id"],as_of_at=T0,algorithm="greedy")
    assert optimize_budget(*args,**kwargs).results["greedy"].summary["budget_compliant"]
    assert len(run_what_if(*args,**kwargs).summary) == 3
    assert result.audit["dispatch_authorized"] is False


def test_no_requests_preserves_roster_and_withholds(data):
    data["exposures"] = data["exposures"].iloc[3:]
    data["historical_features"] = data["historical_features"].iloc[3:]
    data["order_lines"] = data["order_lines"].iloc[:0]
    data["cohort"] = replace(data["cohort"], assignment_count=1)
    result = estimate(data)
    assert result.status == "withheld" and len(result.audit["customers"]) == 4
    assert result.audit["population"]["exact"] == dict.fromkeys(estimator.COMPONENTS)


def test_development_provenance_and_detectable_overlap(data):
    dev = data["historical_features"].assign(customer_id=["dev1","dev2","dev3","dev4"],
                                             dataset_id="dev", raw_database_sha256="d"*64)
    result = estimate(data | {"development_features": dev})
    assert result.audit["development"] == dict(customers=4, dataset_id="dev", raw_database_sha256="d"*64)
    dev.loc[0,"customer_id"] = "001"
    with pytest.raises(InputValidationError, match="overlap"):
        estimate(data | {"development_features": dev})


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("digits", [0, 3, 4, 9])
def test_numpy_precision_matches_python(data, dtype, digits):
    settings = dict(probability_places=digits, discount_places=0, send_places=2, decimal_places=max(2, digits))
    ref = estimate(data, **settings)
    result = estimate(data, **{k: dtype(v) for k, v in settings.items()})
    pd.testing.assert_frame_equal(result.cost_inputs, ref.cost_inputs)
    assert result.audit == ref.audit and result.cost_metadata == ref.cost_metadata
    for term in result.audit["applied"]["population"].values():
        error = Fraction(term["output"]) - Fraction(term["exact"])
        assert error == Fraction(term["error"]) and error >= 0


@pytest.mark.parametrize("key", ["historical_features", "exposures", "target_features"])
@pytest.mark.parametrize("value", ["1.00000000000000000000000000001", "NaN", "Infinity", "0", "-1"])
def test_exact_original_unit_required(data, key, value):
    for name in ("historical_features", "exposures", "target_features"):
        data[name]["amount_unit"] = "1.00"
    data[key].loc[0, "amount_unit"] = value
    with localcontext() as ctx:
        ctx.prec = 5
        with pytest.raises(InputValidationError, match="amount_unit"):
            estimate(data)


@pytest.mark.parametrize("unit", ["1", "1.00000000000000000000000000001"])
def test_equivalent_units_keep_exact_value(data, unit):
    for key, suffix in (("historical_features", ""), ("exposures", "0"), ("target_features", "00")):
        data[key]["amount_unit"] = (unit if "." in unit else unit + ".0") + suffix
    with localcontext() as ctx:
        ctx.prec = 4
        result = estimate(data)
    assert all(Decimal(v) == Decimal(unit) for v in result.cost_inputs.amount_unit)
    assert Decimal(result.audit["target"]["amount_unit"]) == Decimal(unit)


@pytest.mark.parametrize("amount", [True, np.bool_(True), 2, ".1", -1, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_every_paid_minor_checked_before_grouping(data, amount, reverse):
    lines = data["order_lines"]
    lines["unit_price_minor"], lines["quantity"] = [1, 0], 1
    lines["paid_minor"] = pd.Series([1, amount], dtype=object)
    data["exposures"].loc[0, "outcome_redemption_discount_cost_minor"] = 0
    if reverse:
        data["order_lines"] = lines.iloc[::-1]
    if type(amount) is int and amount == 1:
        assert estimate(data).audit["population"]["discounts"] == 0
    else:
        with pytest.raises(InputValidationError):
            estimate(data)


def test_unissued_rate_can_be_missing(data):
    data["exposures"].loc[[2, 3], "discount_rate"] = None
    stats = estimate(data).audit["population"]
    assert (stats["requests"], stats["sent"], stats["charges"]) == (3, 2, 5)
    data["exposures"].loc[0, "discount_rate"] = None
    with pytest.raises(InputValidationError):
        estimate(data)


def test_rare_redemption_rounding_error_is_visible(data):
    for name in ("historical_features", "exposures"):
        extra = pd.concat([data[name].iloc[[1]]] * 997, ignore_index=True)
        ids = [f"extra-{i}" for i in range(len(extra))]
        extra["customer_id"] = extra["snapshot_id"] = ids
        if name == "exposures":
            extra["request_id"] = extra["coupon_id"] = ids
        data[name] = pd.concat([data[name], extra], ignore_index=True)
    data["cohort"] = replace(data["cohort"], assignment_count=1001)
    data["order_lines"]["unit_price_minor"] = [10**12, 20]
    data["order_lines"]["paid_minor"] = 19 * 10**11 + 20
    data["exposures"].loc[0, "outcome_redemption_discount_cost_minor"] = 10**11
    for digits, error in [(3, ".001"), (0, "99900000000.001")]:
        result = estimate(data, probability_places=np.uint8(digits))
        detail = result.audit["applied"]["population"]["expected_cost"]
        assert Fraction(detail["exact"]) == Fraction("100000001.999")
        assert Fraction(detail["output"]) - Fraction(detail["exact"]) == Fraction(detail["error"]) == Fraction(error)
