import numpy as np
import pandas as pd
import pytest

from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.irl import fit_causal_shopping as causal
from simulate.v4.irl.source_observed_features import (
    CHECKOUT_PROXY_FEATURES, UNKNOWN_INITIAL_CART_FEATURES, UNOBSERVED_CHANNEL_FEATURES,
    build_source_observations,
)
from simulate.v4.time_spec import weekly_metadata


T0 = pd.Timestamp("2020-01-01", tz="UTC")


def frame(events):
    return pd.DataFrame([{"user_id": "u", "session": session, "at": T0 + pd.Timedelta(seconds=seconds),
        "action": action, "product_id": product, "price_minor": price, "source_event_id": f"event:{index:05d}"}
        for index, (session, seconds, action, product, price) in enumerate(events)])


def extract(rows, metadata=None):
    metadata = {"observation_start_at": (T0 - pd.Timedelta(days=200)).isoformat(),
                "observation_end_at": (T0 + pd.Timedelta(days=365)).isoformat(), **(metadata or {})}
    records = causal.build_demonstrations(rows, pd.Timestamp(metadata["observation_end_at"]))["records"]
    values, audit = build_source_observations(rows, records, metadata=metadata)
    return pd.DataFrame(values, columns=EXPANDED_FEATURES), records, audit


def test_checkout_payload_is_consumed_after_prediction_and_unsupported_is_unknown():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "cart", "A", 100.),
                  (1, 2, "purchase", "A", 100.), (1, 2, "purchase", "B", 300.),
                  (1, 3, "view", "B", 400.), (2, 86400, "view", "C", 500.)])
    values, records, audit = extract(rows)
    assert values.shape == (len(records), 39) and audit["alignment_exact"]
    assert values.loc[:, [*UNOBSERVED_CHANNEL_FEATURES, *UNKNOWN_INITIAL_CART_FEATURES]].isna().all().all()
    # Context for the purchase sees earlier events, not the purchase's own two rows.
    purchase = records.index[(records.session == 1) & records.action.eq(3)][0]
    assert values.loc[purchase, "purchase_count_30d"] == values.loc[purchase, "purchased_qty_30d"] == 0
    # Strict < excludes all consumed ties, without moving the rolling window.
    after_purchase = purchase + 1
    assert values.loc[after_purchase, "purchase_count_30d"] == 0
    ending = records.index[(records.session == 1) & records.action.eq(4)][0]
    assert values.loc[ending, "purchase_count_30d"] == 1
    assert values.loc[ending, "purchased_qty_30d"] == 2
    assert values.loc[ending, "net_spend_30d"] == 400
    assert pd.isna(values.loc[ending, "mean_session_duration_minutes_30d"])
    later = records.index[(records.session == 2) & records.step.eq(0)][0]
    assert values.loc[later, "mean_session_duration_minutes_30d"] == 3 / 60
    assert values.loc[later, "last_session_min_viewed_price"] == 100
    assert values.loc[later, "last_session_max_viewed_price"] == 400
    assert audit["checkout_groups"] == 1 and audit["collapsed_purchase_rows"] == 1
    assert sum(audit["supported_mask"]) == 28


def test_completion_is_unavailable_at_the_exact_thirty_minute_boundary():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "view", "B", 200.),
                  (2, 1801, "view", "C", 300.), (2, 1802, "cart", "C", 300.),
                  (2, 1803, "view", "C", 400.)])
    values, records, _ = extract(rows)
    exact = records.index[records.session.eq(2) & records.step.eq(0)][0]
    later = records.index[records.session.eq(2) & records.step.eq(2)][0]
    assert pd.isna(values.loc[exact, "last_session_min_viewed_price"])
    assert values.loc[later, "last_session_min_viewed_price"] == 100
    assert values.loc[later, "last_session_max_viewed_price"] == 200
    assert values.loc[later, "mean_session_duration_minutes_30d"] == 1 / 60


def test_future_target_prices_and_sessions_do_not_change_earlier_covariates():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "cart", "A", 100.),
                  (1, 2, "purchase", "A", 100.), (1, 3, "view", "B", 300.),
                  (2, 86400, "view", "Z", 999.)])
    before, records, _ = extract(rows)
    altered = rows.copy()
    altered.loc[altered.index >= 2, "price_minor"] = 900000.
    after, changed, _ = extract(altered)
    np.testing.assert_array_equal(before.iloc[:3].to_numpy(), after.iloc[:3].to_numpy())
    pd.testing.assert_frame_equal(records.iloc[:3], changed.iloc[:3])
    altered.loc[4, "at"] += pd.Timedelta(days=50)
    after, _, _ = extract(altered)
    np.testing.assert_array_equal(before.iloc[:3].to_numpy(), after.iloc[:3].to_numpy())


def test_weekly_and_v5_windows_keep_exact_lower_boundaries():
    day = 86400
    rows = frame([(1, -57 * day, "purchase", "A", 100.), (2, -56 * day, "purchase", "B", 100.),
                  (3, -29 * day, "view", "C", 100.), (4, -28 * day, "purchase", "D", 100.),
                  (5, 0, "view", "E", 100.)])
    legacy, records, legacy_audit = extract(rows)
    weekly, _, audit = extract(rows, weekly_metadata())
    final = records.index[-1]
    assert weekly.loc[final, "purchase_count_30d"] == legacy.loc[final, "purchase_count_30d"] == 1
    assert weekly.loc[final, "orders_delta_30d"] == 0
    assert legacy.loc[final, "orders_delta_30d"] == -1
    assert weekly.loc[final, "view_count_30d"] == 0
    assert legacy.loc[final, "view_count_30d"] == 1
    assert audit["time_windows_days"] == {"recent": 28, "delta": 56, "visit_gap": 91}
    assert legacy_audit["time_windows_days"] == {"recent": 30, "delta": 60, "visit_gap": 90}


def test_unknown_item_price_does_not_become_zero_spend_and_proxy_can_be_disabled():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "purchase", "A", 100.),
                  (1, 1, "purchase", "B", np.nan), (1, 2, "view", "A", 100.)])
    values, records, audit = extract(rows)
    final = records.index[-1]
    assert values.loc[final, "purchased_qty_30d"] == 2
    assert values.loc[final, "unique_purchase_products_30d"] == 2
    assert pd.isna(values.loc[final, "net_spend_30d"])
    assert audit["checkout_groups_with_unknown_price"] == 1
    strict, _, strict_audit = extract(rows, {"source_checkout_proxy": False})
    assert strict.loc[:, CHECKOUT_PROXY_FEATURES].isna().all().all()
    assert strict.loc[final, "unique_purchase_products_30d"] == 2
    assert sum(strict_audit["supported_mask"]) == 17


def test_capture_coverage_and_alignment_are_not_invented():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "view", "B", 200.)])
    values, records, _ = extract(rows, {"observation_start_at": T0.isoformat()})
    assert values.session_count_30d.isna().all()
    broken = records.copy()
    broken.loc[1, "at"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="alignment mismatch"):
        build_source_observations(rows, broken)


def test_unobserved_source_actions_prices_and_subday_clocks_are_missing_not_zero():
    rows = frame([(1, 0, "view", "merchant:A", np.nan),
                  (1, 0, "purchase", "merchant:A", np.nan),
                  (2, 86400, "view", "merchant:B", np.nan)])
    values, records, audit = extract(rows, {"observed_actions": ["view", "purchase"],
        "observed_prices": False, "clock_resolution": "day"})
    for name in ("cart_adds_per_nonpurchase_session_30d", "cart_removes_per_nonpurchase_session_30d",
                 "net_spend_30d", "mean_net_order_value_30d", "net_spend_delta_30d",
                 "mean_net_order_value_delta_30d", "last_session_min_viewed_price",
                 "last_session_max_viewed_price", "mean_session_duration_minutes_30d", "last_session_start_hour"):
        assert values[name].isna().all()
        assert not audit["supported_mask"][EXPANDED_FEATURES.index(name)]
    assert values.loc[records.index[-1], "purchase_count_30d"] == 1
    assert values.loc[records.index[-1], "session_count_30d"] == 1
    assert sum(audit["feature_counts"].values()) == 39
    with pytest.raises(ValueError, match="outside.*support"):
        extract(rows, {"observed_actions": ["view"]})


def test_electronics_missing_remove_is_not_an_observed_zero_remove_rate():
    rows = frame([(1, 0, "view", "A", 100.), (1, 1, "cart", "A", 100.),
                  (2, 86400, "view", "B", 200.)])
    values, records, audit = extract(rows, {"observed_actions": ["view", "cart", "purchase"]})
    final = records.index[-1]
    assert values.loc[final, "cart_adds_per_nonpurchase_session_30d"] == 1
    assert values.cart_removes_per_nonpurchase_session_30d.isna().all()
    assert "cart_removes_per_nonpurchase_session_30d" in audit["unsupported_feature_names"]
