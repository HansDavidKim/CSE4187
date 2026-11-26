"""Independent v5-contract fixtures; no publisher, real model, or dispatch."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pandas as pd
import pytest

from final_strategy.retention_strategy import (SegmentPolicy, HistoryPolicy, HistoryContext, default_segment_policies,
    build_retention_strategy, audit_observed_coupon_order, COUPON_COLUMNS, REDEMPTION_COLUMNS, COST_INPUT_COLUMNS)
from final_strategy.segmentation import segment_customers
from final_strategy.budget import Budget, optimize_budget
from final_strategy.what_if import run_what_if
from final_strategy.validation import CONTEXT_COLUMNS, InputValidationError
from final_strategy.tests.test_segmentation import make_inputs, eligibility_for, T0
from final_strategy.budget import CostMetadata


def setup(records=None, upstream=None):
    inputs = make_inputs(records or [("001", .8, .2, 100), ("002", .2, 0, 200), ("003", .8, -.1, 100)])
    seg = segment_customers(inputs, eligibility=eligibility_for(inputs, upstream) if upstream else None)
    eligibility = seg.customers[list(CONTEXT_COLUMNS)].copy()
    eligibility["business_eligible"] = pd.Series([True] * len(eligibility), dtype=object)
    eligibility["business_reason"] = None
    eligibility["history_complete"] = True
    costs = seg.customers.loc[seg.customers.uplift_score.gt(0), list(CONTEXT_COLUMNS)].copy()
    costs = costs.reset_index(drop=True)
    costs["send_cost"], costs["redemption_probability"], costs["expected_discount_if_redeemed"] = "2", ".25", "10"
    costs["source_id"], costs["source_version"], costs["source_as_of_at"] = "explicit-dummy", "v1", T0
    return [seg, eligibility, pd.DataFrame(columns=COUPON_COLUMNS), pd.DataFrame(columns=REDEMPTION_COLUMNS),
            costs, CostMetadata("expected-send-plus-redemption-28d", "v1", 3, T0)]


def run(bundle=None, **kwargs):
    bundle = setup() if bundle is None else bundle
    options = dict(dataset_id="dummy-segmentation", as_of_at=T0, currency="SOURCE_UNSPECIFIED", amount_unit=".01",
                   history_context=HistoryContext("dummy-segmentation", "a" * 64, T0, "explicit-history", "v1", T0),
                   history_policy=HistoryPolicy(False, False, pd.Timedelta(0)))
    options.update(kwargs)
    return build_retention_strategy(*bundle, **options)


def coupon(sent=None):
    sent = T0 - pd.Timedelta(days=2) if sent is None else sent
    return pd.DataFrame([dict(coupon_id="C1", customer_id="001", sent_at=sent,
        expires_at=sent + pd.Timedelta(days=28), discount_rate=.1, policy_id="v5-fixed-28d", send_cost_minor=100)])


def test_development_positive_uplift_and_exact_cost():
    result = run()
    rows = result.evaluation.set_index("customer_id")
    assert rows.budget_candidate.to_dict() == {"001": True, "002": False, "003": False}
    assert result.costs.expected_cost.tolist() == [Decimal("4.50")]
    assert set(result.costs) == {"dataset_id", "customer_id", "as_of_at", "snapshot_id", "raw_database_sha256", "currency", "amount_unit", "expected_cost"}
    assert "actual_sent" not in rows and "sent_at" not in rows
    assert rows.loc["001", "planned_sent_at"] == T0


def test_current_contract_metadata_preserves_default_validity_and_no_dispatch():
    result = run()
    assert result.metadata["contract"] == (
        "v5 default 28-day coupon contract; declared expected costs; no actual dispatch"
    )
    assert result.metadata["history_policy"]["expected_validity_days"] == 28
    assert result.metadata["actual_dispatch_performed"] is False


def test_twelve_segment_rules_and_no_reclassification():
    records = [(f"{i:03}", c, u, 1000 if j < 2 else 100)
               for i, (c, u, j) in enumerate((c,u,j) for c,u in product((.2,.8),(-.1,0,.2)) for j in range(10))]
    bundle = setup(records)
    assert bundle[0].customers.segment_id.nunique() == 12
    result = run(bundle)
    columns = ["customer_id", "segment_id", "clv_band", "is_high_clv", "priority_score", "priority_rank"]
    wanted = bundle[0].customers.loc[bundle[0].customers.uplift_score.gt(0), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(result.budget_inputs.customers[columns].reset_index(drop=True), wanted)
    assert result.budget_inputs.summary.customer_count.sum() == len(wanted)


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_policy_coverage(kind):
    policies = default_segment_policies()
    policies = policies[:-1] if kind == "missing" else (*policies[:-1], policies[0])
    with pytest.raises(InputValidationError, match="twelve"): run(policies=policies)


def test_custom_policy_disable_threshold_and_delayed_plan():
    bundle = setup()
    identifier = bundle[0].customers.loc[bundle[0].customers.customer_id.eq("001"), "segment_id"].iloc[0]
    policies = [replace(p, send_delay="2h") if p.segment_id == identifier else p for p in default_segment_policies()]
    result = run(bundle, policies=policies)
    row = result.evaluation.set_index("customer_id").loc["001"]
    assert row.planned_sent_at == T0 + pd.Timedelta(hours=2) and row.requires_eligibility_recheck
    bundle[4] = bundle[4].iloc[:0]
    disabled = [replace(p, allow_candidate=False, exclusion_reason="explicit_policy") if p.segment_id == identifier else p for p in policies]
    assert run(bundle, policies=disabled).budget_inputs.customers.empty
    threshold = [replace(p, min_uplift=".2") if p.segment_id == identifier else p for p in policies]
    assert run(bundle, policies=threshold).budget_inputs.customers.empty


@pytest.mark.parametrize("complete,mode,expected", [(False,"unknown",None),(False,"exclude",False),(True,"unknown",True)])
def test_history_completeness(complete, mode, expected):
    bundle = setup();bundle[1]["history_complete"] = complete
    if not complete: bundle[4] = bundle[4].iloc[:0]
    result = run(bundle, history_policy=HistoryPolicy(False, False, "0d", incomplete_history=mode))
    value = result.evaluation.set_index("customer_id").loc["001", "eligible_for_coupon"]
    assert pd.isna(value) if expected is None else value == expected


def test_incomplete_history_error():
    bundle = setup();bundle[1]["history_complete"] = False
    with pytest.raises(InputValidationError, match="incomplete history"):
        run(bundle, history_policy=HistoryPolicy(False, False, "0d", incomplete_history="error"))


@pytest.mark.parametrize("business", [False, None])
def test_business_eligibility_is_separate_from_policy(business):
    bundle = setup();bundle[1].loc[0, "business_eligible"] = business;bundle[1].loc[0,"business_reason"] = "explicit_business_state"
    bundle[4] = bundle[4].iloc[:0]
    row = run(bundle).evaluation.set_index("customer_id").loc["001"]
    assert row.policy_candidate and not row.budget_candidate
    assert row.eligibility_reason == "explicit_business_state"


@pytest.mark.parametrize("state,expected", [("active",False),("expired",True),("redeemed",True),("future_redemption",False),("future_coupon",True)])
def test_observable_coupon_states(state, expected):
    bundle = setup()
    sent = T0 - pd.Timedelta(days=28) if state == "expired" else T0 if state == "future_coupon" else T0 - pd.Timedelta(days=2)
    bundle[2] = coupon(sent)
    if state in ("redeemed","future_redemption"):
        at = T0 - pd.Timedelta(days=1) if state == "redeemed" else T0 + pd.Timedelta(days=1)
        bundle[3] = pd.DataFrame([dict(coupon_id="C1", customer_id="001", purchased_at=at)])
    if not expected: bundle[4] = bundle[4].iloc[:0]
    result = run(bundle, history_policy=HistoryPolicy(True, False, "0d"))
    assert bool(result.evaluation.set_index("customer_id").loc["001", "budget_candidate"]) == expected


def test_previously_redeemed_and_minimum_interval_are_explicit():
    bundle = setup();bundle[2] = coupon()
    bundle[3] = pd.DataFrame([dict(coupon_id="C1",customer_id="001",purchased_at=T0-pd.Timedelta(days=1))])
    bundle[4] = bundle[4].iloc[:0]
    assert run(bundle, history_policy=HistoryPolicy(False, True, "0d")).budget_inputs.customers.empty
    assert run(bundle, history_policy=HistoryPolicy(False, False, "3d")).budget_inputs.customers.empty
    bundle[4] = setup()[4]
    assert len(run(bundle, history_policy=HistoryPolicy(False,False,"2d")).budget_inputs.customers) == 1


@pytest.mark.parametrize("change", ["v4_validity", "duplicate", "double_use", "wrong_owner", "expired_use"])
def test_history_contract_errors(change):
    bundle = setup();bundle[2] = coupon()
    if change == "v4_validity":bundle[2]["expires_at"] += pd.Timedelta(days=2)
    if change == "duplicate":bundle[2] = pd.concat([bundle[2],bundle[2]])
    if change in ("double_use","wrong_owner","expired_use"):
        bundle[3] = pd.DataFrame([dict(coupon_id="C1",customer_id="other" if change=="wrong_owner" else "001",
                                      purchased_at=T0+pd.Timedelta(days=28) if change=="expired_use" else T0-pd.Timedelta(days=1))])
        if change == "double_use":bundle[3] = pd.concat([bundle[3],bundle[3]])
    with pytest.raises(InputValidationError):run(bundle)


@pytest.mark.parametrize("field,value", [("send_cost",None),("send_cost",-1),("redemption_probability",None),
    ("redemption_probability","1.01"),("redemption_probability",True),("expected_discount_if_redeemed",np.inf),
    ("expected_discount_if_redeemed",None),("amount_unit","1"),("currency","KRW"),
    ("source_as_of_at",T0+pd.Timedelta(seconds=1)),("raw_database_sha256","b"*64)])
def test_cost_inputs_rejected(field,value):
    bundle=setup();bundle[4][field]=value
    with pytest.raises(InputValidationError):run(bundle)


def test_expected_cost_precision_and_no_paid_amount_input():
    bundle=setup();bundle[4]["redemption_probability"]=".00001"
    with pytest.raises(InputValidationError,match="represent exactly"):run(bundle)
    bundle=setup();bundle[4]["paid_minor"]=999
    with pytest.raises(InputValidationError,match="exactly"):run(bundle)
    with localcontext() as context:
        context.prec=2
        assert run().costs.expected_cost.iloc[0] == Decimal("4.50")


@pytest.mark.parametrize("case", ["dataset","snapshot","sha","time"])
def test_context_mixing_rejected(case):
    bundle=setup()
    column,value={"dataset":("dataset_id","other"),"snapshot":("snapshot_id","other"),
                  "sha":("raw_database_sha256","b"*64),"time":("as_of_at",T0+pd.Timedelta(hours=1))}[case]
    bundle[1].loc[0,column]=value
    with pytest.raises(InputValidationError):run(bundle)


def test_budget_what_if_compatibility_upstream_exclusion_and_originals():
    bundle=setup(upstream={"003":"original_exclusion"});original=deepcopy(bundle)
    result=run(bundle)
    assert result.budget_inputs.excluded.set_index("customer_id").loc["003","exclusion_reason"] == "original_exclusion"
    assert result.budget_inputs.metadata["retention_source_metadata"] == original[0].metadata
    budget=optimize_budget(result.budget_inputs,result.costs,Budget(10,"SOURCE_UNSPECIFIED",".01"),result.cost_metadata,
                           dataset_id="dummy-segmentation",as_of_at=T0)
    assert budget.results["greedy"].summary["selected_customer_count"] == 1
    assert len(run_what_if(result.budget_inputs,result.costs,Budget(10,"SOURCE_UNSPECIFIED",".01"),result.cost_metadata,
                          dataset_id="dummy-segmentation",as_of_at=T0,algorithm="greedy").summary) == 3
    for before,after in zip(original[1:5],bundle[1:5]):pd.testing.assert_frame_equal(before,after)
    pd.testing.assert_frame_equal(original[0].customers,bundle[0].customers)
    assert original[0].metadata == bundle[0].metadata


def test_empty_candidates_do_not_impute_zero_cost():
    bundle=setup([("001",.5,0,100)])
    result=run(bundle)
    assert result.budget_inputs.customers.empty and result.costs.empty
    assert result.cost_estimates.empty


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_cost_customer_set_is_exact(change):
    bundle=setup()
    if change == "missing": bundle[4] = bundle[4].iloc[:0]
    elif change == "duplicate": bundle[4] = pd.concat([bundle[4],bundle[4]])
    else:
        extra=bundle[4].copy();extra["customer_id"]="unexpected"
        bundle[4]=pd.concat([bundle[4],extra])
    with pytest.raises(InputValidationError):run(bundle)


def test_explicit_zero_cost_is_valid_but_precision_loss_is_rejected():
    bundle=setup();bundle[4]["send_cost"]="0";bundle[4]["redemption_probability"]="0"
    assert run(bundle).costs.expected_cost.iloc[0] == 0
    bundle=setup();bundle[4]["redemption_probability"]="1e-1000"
    with pytest.raises(InputValidationError,match="exact computation precision"):run(bundle)


@pytest.mark.parametrize("field", ["as_of_at", "raw_database_sha256"])
def test_source_cohort_mixing_is_rejected(field):
    bundle=setup()
    bundle[0].customers.loc[1,field]=T0+pd.Timedelta(hours=1) if field=="as_of_at" else "b"*64
    with pytest.raises(InputValidationError):run(bundle)


def test_history_provenance_and_knowledge_time_are_required():
    context=HistoryContext("other","a"*64,T0,"history","v1",T0)
    with pytest.raises(InputValidationError,match="mismatch"):run(history_context=context)
    context=replace(context,dataset_id="dummy-segmentation",known_at=T0+pd.Timedelta(seconds=1))
    with pytest.raises(InputValidationError,match="future history"):run(history_context=context)


@pytest.mark.parametrize("declared_days", [30, None])
def test_other_history_validity_requires_explicit_declaration(declared_days):
    bundle=setup();bundle[2]=coupon();bundle[2]["expires_at"] += pd.Timedelta(days=2)
    result=run(bundle,history_policy=HistoryPolicy(False,False,"0d",expected_validity_days=declared_days))
    assert len(result.budget_inputs.customers) == 1  # compatibility declaration, not v5 integration


def test_observed_v5_highest_unit_tie_half_up_and_no_double_discount():
    items=pd.DataFrame({"line_no":[1,2],"quantity":[5,2],"unit_price_minor":[105,250]})
    result=audit_observed_coupon_order(items,purchase_at=T0-pd.Timedelta(days=1),as_of_at=T0,
                                      actual_discount_minor=25,actual_paid_minor=1000)
    assert result == dict(line_no=2,gross_minor=1025,discount_minor=25,paid_minor=1000)
    items["unit_price_minor"]=[255,255]
    assert audit_observed_coupon_order(items,purchase_at=T0-pd.Timedelta(days=1),as_of_at=T0,
                                       actual_discount_minor=26,actual_paid_minor=1759)["line_no"] == 1
    with pytest.raises(InputValidationError):
        audit_observed_coupon_order(items,purchase_at=T0,as_of_at=T0,actual_discount_minor=26,actual_paid_minor=1759)
    with pytest.raises(InputValidationError):
        audit_observed_coupon_order(items,purchase_at=T0-pd.Timedelta(days=1),as_of_at=T0,actual_discount_minor=26,actual_paid_minor=1733)
