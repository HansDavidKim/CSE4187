from copy import deepcopy

import pandas as pd
import pytest

from simulate.v4.data.loader import load_tables
from simulate.v4.features import EXPANDED_FEATURES, build_features
from simulate.v4.time_spec import feature_window_days, weekly_calendar, weekly_metadata
from test_v4_features_sqlite import T0, dataset


def _diagnostics(tables, edit_reason="scheduled_next_start"):
    return tables["sessions"][["customer_id", "session_id"]].assign(
        termination_reason=["causal_policy_end", edit_reason],
        truncated_session=[False, edit_reason != "causal_policy_end"])


def test_public39_keeps_visits_and_actions_but_excludes_censored_completion(dataset):
    tables = load_tables(dataset)
    before = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
    tables["session_diagnostics"] = _diagnostics(tables)
    after = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
    assert len(EXPANDED_FEATURES) == 39
    for name in ("session_count_30d", "view_count_30d", "purchase_count_30d", "net_spend_30d"):
        assert after[name] == before[name]
    assert after.last_session_max_viewed_price == 200
    assert pd.isna(after.cart_adds_per_nonpurchase_session_30d)
    assert before.cart_adds_per_nonpurchase_session_30d == 1


def test_native_end_is_available_strictly_after_thirty_minutes(dataset):
    tables = load_tables(dataset)
    tables["session_diagnostics"] = _diagnostics(tables, "causal_policy_end")
    boundary = pd.Timestamp("2024-02-20T10:40:00Z")
    exact = build_features(tables, as_of_at=boundary).set_index("customer_id").loc["u"]
    later = build_features(tables, as_of_at=boundary + pd.Timedelta(nanoseconds=1)).set_index("customer_id").loc["u"]
    assert exact.last_session_max_viewed_price == 200
    assert pd.isna(later.last_session_max_viewed_price)
    assert pd.isna(exact.cart_adds_per_nonpurchase_session_30d)
    assert later.cart_adds_per_nonpurchase_session_30d == 1
    invalid = deepcopy(tables)
    invalid["session_diagnostics"] = invalid["session_diagnostics"].iloc[:1]
    with pytest.raises(ValueError, match="cover every observed visit"):
        build_features(invalid, as_of_at=boundary)


def test_weekly_windows_preserve_names_and_exclude_day29(dataset):
    tables = load_tables(dataset)
    old = T0 - pd.Timedelta(days=29)
    sessions = tables["sessions"]
    tables["sessions"] = pd.concat([sessions, pd.DataFrame([dict(
        session_id="old", customer_id="u", started_at=old, ended_at=old + pd.Timedelta(minutes=1))])], ignore_index=True)
    event = tables["events"].iloc[[0]].copy()
    event["event_id"], event["session_id"], event["event_at"] = 99, "old", old
    tables["events"] = pd.concat([tables["events"], event], ignore_index=True)
    legacy = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
    tables["_metadata"].update(weekly_metadata())
    weekly = build_features(tables, as_of_at=T0).set_index("customer_id").loc["u"]
    assert legacy.view_count_30d == weekly.view_count_30d + 1
    assert list(weekly[list(EXPANDED_FEATURES)].index) == list(EXPANDED_FEATURES)
    assert feature_window_days({}) == (30, 60, 90)
    assert feature_window_days(tables["_metadata"]) == (28, 56, 91)
    first, middle, last = weekly_calendar("2024-01-01T00:00:00+09:00")
    assert (middle-first).days == (last-middle).days == 364
