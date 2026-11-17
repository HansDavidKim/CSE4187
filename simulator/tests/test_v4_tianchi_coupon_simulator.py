"""Actual coupon checkout from synthetic response counts, without production data."""
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_v4_checkout_simulator import checkout_fixture, _customers
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, discount_minor, validate_raw_dataset


def _config(checkout_fixture, folder, *, redeemed=4, delay=0, legacy_use=0., first_action="view", sessions=4):
    config, model = copy.deepcopy(checkout_fixture)
    folder.mkdir(parents=True, exist_ok=True)
    for profile in model["profiles"]:
        profile.update(session_first_action_counts={first_action:1}, session_choice_lengths=[1],
                       session_lengths=[1], session_counts_active_30d=[sessions], idle_gap_seconds=[86400.])
    groups = {name:dict(receipts=4,redeemed=redeemed,response_probability=redeemed/4,
                       redemption_delay_days=[delay]*redeemed)
              for name in ("no_observed_purchase","purchased_within_28d","lapsed_purchase")}
    groups["pooled"] = dict(receipts=12,redeemed=3*redeemed,response_probability=redeemed/4,
                            redemption_delay_days=[delay]*(3*redeemed))
    response = dict(schema_version=1,type="tianchi_o2o_descriptive_response",discount_rate=.1,
                    window_days=30,inactivity_days=28,source="synthetic response wiring fixture",groups=groups)
    def write(name, value):
        path=folder/name
        path.write_text(json.dumps(value,allow_nan=False))
        return str(path)
    config["source_simulation"].update(model_path=write("shopping.json",model), return_model_path=None,
        coupon_response_model_path=write("response.json",response))
    config["toy"]["coupon_model_path"]=write("legacy-use.json",{
        "groups":{"baseline":{"coupon_use_probability":legacy_use}}})
    config["toy"]["message_model_path"]=write("message.json",{
        "response_lift":{"positive":0.,"neutral":0.,"negative":0.},"visit_lift_scale":1.,
        "observed":{"pooled":{"visit_logit_shift":0.}}})
    return config


def _run(config, count=12):
    return EcommerceSimulator(config).generate(months=3,seed=17,customers=_customers(count))


def test_coupon_day_histogram_replays_ordered_delays(checkout_fixture, tmp_path):
    from simulate.v4.simulator.coupon_response import sample_coupon_response, validate_response_model

    config = _config(checkout_fixture, tmp_path)
    original = json.loads(Path(config["source_simulation"]["coupon_response_model_path"]).read_text())
    aggregate = copy.deepcopy(original)
    for group in original["groups"].values():
        group["redemption_delay_days"][0] = 9
        group["redemption_delay_days"].sort()
    for name, group in aggregate["groups"].items():
        counts = Counter(original["groups"][name]["redemption_delay_days"])
        group.pop("redemption_delay_days")
        group["redemption_delay_day_counts"] = [counts[day] for day in range(30)]
    validate_response_model(aggregate)
    timestamp = pd.Timestamp("2024-05-01T00:00:00Z")
    for seed in range(16):
        assert sample_coupon_response(original, seed, "customer", timestamp, None) == sample_coupon_response(
            aggregate, seed, "customer", timestamp, None)


def _pre_send_cohorts(result):
    tables=result.tables
    t0=pd.Timestamp(result.metadata["as_of_at"])
    purchase=tables["orders"].merge(tables["events"][["event_id","customer_id","event_at"]],
        left_on="purchase_event_id",right_on="event_id",validate="one_to_one")
    prior=purchase.loc[purchase.event_at.lt(t0)].groupby("customer_id").event_at.max()
    expected={uid:("no_observed_purchase" if uid not in prior else "purchased_within_28d"
                   if t0-prior[uid]<=pd.Timedelta(days=28) else "lapsed_purchase")
              for uid in tables["customers"].customer_id}
    assert tables["coupon_response_diagnostics"].set_index("customer_id").history_cohort.to_dict()==expected


def test_zero_purchase_single_action_users_make_real_coupon_orders_and_draw_once(checkout_fixture,tmp_path,monkeypatch):
    import simulate.v4.simulator.engine as engine
    actual=engine.sample_coupon_response
    calls=[]
    def sample(*args):
        calls.append((args[2],args[3],args[4]))
        return actual(*args)
    monkeypatch.setattr(engine,"sample_coupon_response",sample)
    result=_run(_config(checkout_fixture,tmp_path,legacy_use=0.))
    tables=result.tables
    t0=pd.Timestamp(result.metadata["as_of_at"])
    orders=tables["orders"].merge(tables["events"][["event_id","customer_id","event_at"]],
        left_on="purchase_event_id",right_on="event_id",validate="one_to_one")
    arms=tables["treatments"].set_index("customer_id").treatment
    diag=tables["coupon_response_diagnostics"].set_index("customer_id")
    assert not orders.empty
    assert orders.event_at.ge(t0).all() and orders.customer_id.map(arms).eq(1).all()
    assert orders.coupon_id.notna().all() and orders.coupon_id.is_unique
    assert orders.customer_id.is_unique  # Many opportunities still consume one coupon once.
    assert len(calls)==len(tables["customers"]) and len({call[0] for call in calls})==len(calls)
    assert all(last is None and at==t0 for _,at,last in calls)
    assert diag.loc[orders.customer_id,"promoted_purchase"].all()
    assert diag.loc[orders.customer_id,"baseline_action"].eq("view").all()
    assert set(diag.loc[diag.coupon_used,"purchase_event_id"].astype(int))==set(orders.purchase_event_id)
    assert diag.loc[diag.treatment.eq(0),"response_status"].eq("control").all()
    assert diag.loc[diag.treatment.eq(0),"purchase_event_id"].isna().all()
    _pre_send_cohorts(result)  # Future promoted purchases cannot reclassify the send-time cohort.
    validate_raw_dataset({name:tables[name] for name in RAW_COLUMNS},result.metadata)


def test_tianchi_success_bypasses_legacy_use_probability_without_multiplication(checkout_fixture,tmp_path):
    left=_run(_config(checkout_fixture,tmp_path/"zero",legacy_use=0.))
    right=_run(_config(checkout_fixture,tmp_path/"one",legacy_use=1.))
    assert not left.tables["orders"].empty
    for name in RAW_COLUMNS:
        pd.testing.assert_frame_equal(left.tables[name],right.tables[name])
    pd.testing.assert_frame_equal(left.tables["coupon_response_diagnostics"],right.tables["coupon_response_diagnostics"])


def test_zero_response_never_uses_legacy_gate_even_for_natural_purchases(checkout_fixture,tmp_path):
    result=_run(_config(checkout_fixture,tmp_path,redeemed=0,legacy_use=1.,first_action="purchase"))
    tables=result.tables
    assert not tables["orders"].empty
    assert tables["orders"].coupon_id.isna().all() and tables["orders"].discount_minor.eq(0).all()
    assert not tables["events"].event_type.eq("coupon_usage").any()
    diag=tables["coupon_response_diagnostics"]
    assert not diag.response_selected.any() and not diag.coupon_used.any() and not diag.promoted_purchase.any()
    assert diag.purchase_event_id.isna().all()
    assert diag.loc[diag.treatment.eq(1),"response_status"].eq("no_response").all()
    _pre_send_cohorts(result)


def test_zero_response_does_not_manufacture_a_purchase_for_zero_purchase_users(checkout_fixture,tmp_path):
    result=_run(_config(checkout_fixture,tmp_path,redeemed=0,legacy_use=1.))
    assert result.tables["orders"].empty and result.tables["order_items"].empty
    assert result.tables["events"].event_type.eq("view").all()


def test_sampled_day29_response_waits_for_actual_session_and_preserves_28d_outcome(checkout_fixture,tmp_path):
    result=_run(_config(checkout_fixture,tmp_path,delay=29,sessions=31))
    tables=result.tables
    t0=pd.Timestamp(result.metadata["as_of_at"])
    bought=tables["orders"].merge(tables["events"][["event_id","session_id","event_at"]],
        left_on="purchase_event_id",right_on="event_id",validate="one_to_one")
    assert not bought.empty
    assert bought.event_at.ge(t0+pd.Timedelta(days=29)).all()
    assert bought.event_at.lt(t0+pd.Timedelta(days=30)).all()
    visits=tables["sessions"].set_index("session_id")
    assert bought.event_at.eq(bought.session_id.map(visits.started_at)).all()
    assert tables["labels"].future_revenue_28d.eq(0).all()
    assert tables["labels"].future_revenue_30d.gt(0).any()
    used=tables["coupon_response_diagnostics"].loc[lambda frame:frame.coupon_used]
    assert used.source_delay_days.eq(29).all()
    assert pd.to_datetime(used.opportunity_at,utc=True).ge(pd.to_datetime(used.response_ready_at,utc=True)).all()
    assert bought.coupon_id.is_unique
    validate_raw_dataset({name:tables[name] for name in RAW_COLUMNS},result.metadata)


@pytest.mark.parametrize("delay", [28, 29])
def test_coupon_expiry_blocks_response_at_or_after_day_28(checkout_fixture, tmp_path, delay):
    config = _config(checkout_fixture, tmp_path, delay=delay, sessions=31)
    config["source_simulation"]["coupon_validity_days"] = 28
    result = _run(config)
    t0 = pd.Timestamp(result.metadata["as_of_at"])
    expiry = t0 + pd.Timedelta(days=28)
    assert result.tables["sessions"].started_at.ge(expiry).any()
    assert result.tables["coupons"].expires_at.eq(expiry).all()
    assert result.tables["orders"].empty
    treated = result.tables["coupon_response_diagnostics"].query("treatment == 1")
    assert treated.response_selected.all()
    assert treated.response_ready_at.ge(expiry).all()
    assert treated.response_status.eq("no_eligible_opportunity").all()


def test_selected_response_without_opportunity_before_expiry_creates_no_order(checkout_fixture,tmp_path):
    result=_run(_config(checkout_fixture,tmp_path,delay=29,sessions=1))
    tables=result.tables
    t0=pd.Timestamp(result.metadata["as_of_at"])
    assert tables["sessions"].started_at.ge(t0+pd.Timedelta(days=30)).any()
    assert tables["orders"].empty
    treated=tables["coupon_response_diagnostics"].loc[lambda frame:frame.treatment.eq(1)]
    assert treated.response_selected.all() and not treated.coupon_used.any()
    assert treated.opportunity_at.isna().all() and treated.purchase_event_id.isna().all()
    assert treated.response_status.eq("no_eligible_opportunity").all()


def test_response_noise_is_keyed_by_customer_not_batch_or_global_rng(checkout_fixture,tmp_path):
    config=_config(checkout_fixture,tmp_path,redeemed=2)
    first=_run(config,count=12)
    np.random.seed(987)
    np.random.random(100)
    again=_run(config,count=12)
    for name in first.tables:
        pd.testing.assert_frame_equal(first.tables[name],again.tables[name])
    small=_run(config,count=8)
    columns=["response_probability","response_uniform","response_selected","source_delay_days","history_cohort"]
    large=first.tables["coupon_response_diagnostics"].set_index("customer_id")[columns]
    subset=small.tables["coupon_response_diagnostics"].set_index("customer_id")[columns]
    pd.testing.assert_frame_equal(large.loc[subset.index],subset)


def test_promoted_checkout_reconciles_persistent_cart_and_integer_money(checkout_fixture,tmp_path):
    result=_run(_config(checkout_fixture,tmp_path,first_action="cart",legacy_use=0.))
    tables=result.tables
    assert not tables["orders"].empty
    purchases=tables["events"].set_index("event_id")
    for order in tables["orders"].itertuples(index=False):
        event=purchases.loc[order.purchase_event_id]
        before=Counter(cart_at(tables,event.customer_id,event.event_at,order.purchase_event_id))
        items=tables["order_items"].loc[tables["order_items"].order_id.eq(order.order_id)]
        basket=Counter(dict(zip(items.product_id,items.quantity.astype(int))))
        assert before  # Existing cart participates; this is not only a direct-purchase case.
        expected={pid:quantity-min(quantity,basket[pid]) for pid,quantity in before.items()
                  if quantity-min(quantity,basket[pid])>0}
        assert cart_at(tables,event.customer_id,event.event_at+pd.Timedelta(nanoseconds=1))==expected
        gross=sum(int(row.quantity)*int(row.unit_price_minor) for row in items.itertuples(index=False))
        assert order.gross_minor==gross==355
        assert order.discount_minor==discount_minor(int(items.unit_price_minor.max()),.1)==25
        assert order.paid_minor==330 and order.coupon_id is not None
    validate_raw_dataset({name:tables[name] for name in RAW_COLUMNS},result.metadata)
