import copy
import numpy as np
import pandas as pd
import pytest
from retention.data import Records, normalize, validate, SCHEMA
from retention.features import FeatureBuilder, sequences
from retention.dataset import make_dataset, rolling_folds


def test_order_amount_not_multiplied_and_coupon_time_optional(records):
    f, _, _ = FeatureBuilder(records).build("u", pd.Timestamp("2024-02-02"), pd.Timestamp("2024-01-01"))
    assert f["orders_30d"] == 1 and f["gross_spend_30d"] == 25
    assert f["coupon_uses_30d"] == 1
    assert np.isnan(f["coupon_sends_30d"])
    assert np.isnan(f["mean_purchase_gap_days"])


def test_timestamp_tie_order_invariant(records):
    e = records["events"]
    a, _ = sequences(e, pd.Timestamp("2024-02-02"))
    b, _ = sequences(e.iloc[::-1], pd.Timestamp("2024-02-02"))
    np.testing.assert_array_equal(a, b)


def test_future_changes_do_not_change_past_features(records):
    before = pd.Timestamp("2024-02-02")
    f1, s1, _ = FeatureBuilder(records).build("u", before, pd.Timestamp("2024-01-01"))
    future = records["events"].iloc[[0]].copy()
    future.event_id = "future"; future.event_at = pd.Timestamp("2024-03-01")
    records.tables["events"] = pd.concat([records["events"], future])
    f2, s2, _ = FeatureBuilder(records).build("u", before, pd.Timestamp("2024-01-01"))
    pd.testing.assert_series_equal(pd.Series(f1), pd.Series(f2))
    np.testing.assert_array_equal(s1, s2)


def test_current_session_final_duration_not_visible(records):
    f, _, _ = FeatureBuilder(records).build("u", pd.Timestamp("2024-02-01 00:01:30"), pd.Timestamp("2024-01-01"))
    assert np.isnan(f["duration_minutes_30d"])


def test_exact_boundary_in_label_not_features(records):
    cfg = {"observation_end": "2024-04-01", "horizon_days": 30}
    old = records["sessions"].copy()
    old.session_id = "old"
    old.started_at = pd.Timestamp("2024-01-31")
    old.ended_at = pd.Timestamp("2024-01-31 00:02:00")
    records.tables["sessions"] = pd.concat([old, records["sessions"]])
    meta, x, _, _ = make_dataset(records, cfg, ["2024-02-01"])
    assert meta.y.iloc[0] == 0
    assert x.sessions_30d.iloc[0] == 1
    assert x.purchase_count_30d.iloc[0] == 0


def test_no_immature_labels(records):
    with pytest.raises(ValueError, match="No eligible"):
        make_dataset(records, {"observation_end": "2024-02-15"}, ["2024-02-02"])


def test_duplicate_and_broken_references_fail(records):
    bad = copy.deepcopy(records)
    bad.tables["orders"] = pd.concat([bad["orders"], bad["orders"]])
    with pytest.raises(ValueError, match="duplicate key"):
        validate(bad)
    bad = copy.deepcopy(records); bad.tables["redemptions"].loc[0, "order_id"] = "missing"
    with pytest.raises(ValueError, match="unknown coupon or order"):
        validate(bad)


def test_purged_folds():
    dates = pd.date_range("2024-01-01", periods=30, freq="7D")
    m = pd.DataFrame([{"customer_id": str(u), "as_of": t, "label_end": t + pd.Timedelta(days=30), "y": u % 2} for t in dates for u in range(6)])
    folds, tr, te = rolling_folds(m, "2024-06-01")
    assert len(folds) == 5
    for a, b in folds:
        assert m.loc[a, "label_end"].max() <= m.loc[b, "as_of"].min()
    assert m.loc[tr, "label_end"].max() <= m.loc[te, "as_of"].min()


def test_unknown_customers_and_coupon_owner_fail(records):
    bad = copy.deepcopy(records)
    bad.tables["coupons"].loc[0, "customer_id"] = "missing"
    with pytest.raises(ValueError, match="unknown customer"):
        validate(bad)


def test_null_price_and_partial_optional_timestamp_fail(records):
    bad = copy.deepcopy(records)
    bad.tables["items"].loc[0, "unit_price"] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        validate(bad)
    bad = copy.deepcopy(records)
    bad.tables["coupons"]["sent_at"] = pd.NaT
    with pytest.raises(ValueError, match="optional sent_at"):
        validate(bad)


def test_no_eligible_inference_returns_status(records):
    meta, x, seqs, excluded = make_dataset(records, {}, ["2023-01-01"], labeled=False)
    assert meta.empty and x.empty and seqs == []
    assert excluded.reason.tolist() == ["not_yet_eligible_age"]


def test_customer_session_composite_key(records):
    bad = copy.deepcopy(records)
    customer = pd.DataFrame({"customer_id": ["v"], "signed_up_at": [pd.Timestamp("2024-01-01")]})
    bad.tables["customers"] = pd.concat([bad["customers"], customer])
    duplicate_id = bad["sessions"].copy(); duplicate_id.customer_id = "v"
    bad.tables["sessions"] = pd.concat([bad["sessions"], duplicate_id])
    validate(bad)  # Same session ID under a different customer is valid.
    f, _, _ = FeatureBuilder(bad).build("u", pd.Timestamp("2024-02-02"), pd.Timestamp("2024-01-01"))
    assert f["sessions_30d"] == 1
