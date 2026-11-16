import copy
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from simulate.v4.simulator.outcomes import redefine_outcomes


def outcome_fixture():
    t0 = pd.Timestamp("2025-01-06", tz="UTC")
    sessions = []
    for uid in ("A", "B", "C", "D"):
        sessions.append((uid + "-past", uid, t0 - pd.Timedelta(days=1)))
    sessions += [("A-future", "A", t0), ("B-boundary", "B", t0 + pd.Timedelta(days=28)), ("D-future", "D", t0 + pd.Timedelta(days=28) - pd.Timedelta(nanoseconds=1))]
    times = [t0, t0 + pd.Timedelta(days=27), t0 + pd.Timedelta(days=28), t0 + pd.Timedelta(days=29)]
    events = pd.DataFrame({"event_id": [1, 2, 3, 4], "customer_id": ["A"] * 4, "event_at": times, "event_type": ["purchase"] * 4})
    labels = pd.DataFrame({"customer_id": list("ABCD"), "as_of_at": [t0] * 4, "snapshot_id": [f"{uid}:{t0}" for uid in "ABCD"], "churn": [0, 0, 1, 0], "retained": [1, 1, 0, 1], "future_retained": [1, 1, 0, 1], "future_revenue": [1000, 0, 0, 0], "future_revenue_30d": [1000, 0, 0, 0], "future_revenue_12m": [9000, np.nan, 4000, np.nan], "outcome_end_at": [t0 + pd.Timedelta(days=30)] * 4, "policy_id": ["no_new_coupon", "one_coupon_10pct_30d"] * 2, "definition_id": ["future_no_session_30d_active30d"] * 4})
    tables = {
        "customers": pd.DataFrame({"customer_id": list("ABCD"), "signed_up_at": [t0 - pd.Timedelta(days=100)] * 4}),
        "products": pd.DataFrame({"product_id": ["P"], "category_id": ["skin"], "unit_price_minor": [100]}),
        "sessions": pd.DataFrame(sessions, columns=["session_id", "customer_id", "started_at"]),
        "events": events,
        "orders": pd.DataFrame({"order_id": list("1234"), "purchase_event_id": [1, 2, 3, 4], "paid_minor": [100, 200, 300, 400]}),
        "order_items": pd.DataFrame({"order_id": list("1234"), "product_id": ["P"] * 4, "quantity": [1, 2, 3, 4]}),
        "coupons": pd.DataFrame({"coupon_id": ["coupon-B", "coupon-D"], "customer_id": ["B", "D"], "sent_at": [t0] * 2, "expires_at": [t0 + pd.Timedelta(days=30)] * 2}),
        "treatments": pd.DataFrame({"customer_id": list("ABCD"), "as_of_at": [t0] * 4, "treatment": [0, 1, 0, 1], "assignment_probability": [.5] * 4}),
        "labels": labels,
        "diagnostics": pd.DataFrame({"customer_id": list("ABCD"), "source_profile_id": ["pA", "pB", "unknown", "pD"], "home_persona": ["H", "H", "H", "missing"], "no_coupon_return_probability_30d": [.9, .8, .7, .6], "coupon_return_probability_30d": [.95, .85, .75, .65]}),
    }
    metadata = {"dataset_id": "original30", "definition_id": "future_no_session_30d_active30d", "policy_id": "coupon_once_30d", "observation_start_at": (t0 - pd.Timedelta(days=150)).isoformat(), "observation_end_at": (t0 + pd.Timedelta(days=400)).isoformat(), "expected_control_churn_rate": .25, "observed_churn_rate": .25, "target_churn_rate": .2}
    model = {"profiles": [{"profile_id": "pA", "home_group": "H", "idle_gap_seconds": [1, 28 * 86400]}, {"profile_id": "pB", "home_group": "H", "idle_gap_seconds": [30 * 86400]}, {"profile_id": "pD", "home_group": "missing", "idle_gap_seconds": []}], "home_groups": {"H": {"idle_gap_seconds": [28 * 86400]}}}
    return tables, metadata, model


def test_28day_view_preserves_raw_policy_and_30day_and_clv_targets():
    tables, metadata, model = outcome_fixture()
    original_labels, original_diagnostics, original_metadata = tables["labels"].copy(deep=True), tables["diagnostics"].copy(deep=True), copy.deepcopy(metadata)
    view, info = redefine_outcomes(tables, metadata, shopping_model=model)
    for name in ("customers", "products", "sessions", "events", "orders", "order_items", "coupons", "treatments"):
        assert view[name] is tables[name]
    pd.testing.assert_frame_equal(tables["labels"], original_labels)
    pd.testing.assert_frame_equal(tables["diagnostics"], original_diagnostics)
    assert metadata == original_metadata
    labels = view["labels"].set_index("customer_id")
    assert labels.churn.to_dict() == {"A": 0, "B": 1, "C": 1, "D": 0}
    assert labels.future_revenue_28d.to_dict() == {"A": 300, "B": 0, "C": 0, "D": 0}
    assert labels.future_revenue.to_dict() == labels.future_revenue_28d.to_dict()
    pd.testing.assert_series_equal(view["labels"].future_revenue_30d, original_labels.future_revenue_30d)
    pd.testing.assert_series_equal(view["labels"].future_revenue_12m, original_labels.future_revenue_12m)
    pd.testing.assert_series_equal(view["labels"].policy_id, original_labels.policy_id)
    assert (labels.outcome_end_at == labels.as_of_at + pd.Timedelta(days=28)).all()
    assert set(labels.definition_id) == {"future_no_session_28d_active30d"}
    assert info["dataset_id"] == "original30-28d" and info["parent_dataset_id"] == "original30"
    assert info["generation_window_days"] == 30 and info["outcome_window_days"] == 28 and info["eligibility_window_days"] == 30
    assert info["policy_id"] == "coupon_once_30d" and info["observed_churn_rate"] == .5


def test_analytic_28_probability_uses_exact_gap_tail_and_preserves_p30():
    tables, metadata, model = outcome_fixture()
    view, info = redefine_outcomes(tables, metadata, shopping_model=model)
    diag = view["diagnostics"]
    q = np.array([1 - (2 / 30) * .5, 28 / 30, 28 / 30, 1])
    np.testing.assert_allclose(diag.first_session_within_28d_share, q)
    for prefix in ("no_coupon", "coupon"):
        pd.testing.assert_series_equal(diag[f"{prefix}_return_probability_30d"], tables["diagnostics"][f"{prefix}_return_probability_30d"])
        np.testing.assert_allclose(diag[f"{prefix}_return_probability_28d"], tables["diagnostics"][f"{prefix}_return_probability_30d"] * q)
    expected = np.where(tables["treatments"].treatment.eq(1), diag.coupon_return_probability_28d, diag.no_coupon_return_probability_28d)
    assert info["expected_churn_rate"] == pytest.approx(np.mean(1 - expected))
    assert info["expected_control_churn_rate"] == pytest.approx(1 - diag.no_coupon_return_probability_28d.mean())
    assert info["analytic_outcome_view"]["one_second_fallback_customers"] == 1


def test_immature_view_has_unknown_outcomes_and_preserves_original_columns():
    tables, metadata, model = outcome_fixture()
    metadata["observation_end_at"] = (tables["labels"].as_of_at.iloc[0] + pd.Timedelta(days=27)).isoformat()
    view, info = redefine_outcomes(tables, metadata, shopping_model=model)
    assert view["labels"].churn.isna().all() and view["labels"].future_revenue_28d.isna().all()
    assert view["labels"].is_censored.all()
    assert info["observed_churn_rate"] is None
    pd.testing.assert_series_equal(view["labels"].future_revenue_30d, tables["labels"].future_revenue_30d)


@pytest.mark.parametrize("days", [True, 0, 31, 28.5])
def test_invalid_shorter_observation_horizon_is_rejected(days):
    tables, metadata, model = outcome_fixture()
    with pytest.raises(ValueError, match="outcome_days"):
        redefine_outcomes(tables, metadata, days, model)
