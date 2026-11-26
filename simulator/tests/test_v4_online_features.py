import numpy as np
import pandas as pd
import pytest

from test_v4_features_sqlite import dataset, T0
from simulate.v4.features import EXPANDED_FEATURES, build_features, load_tables
from simulate.v4.simulator.online_features import observed_snapshot


@pytest.mark.parametrize("offset", (-22*86400, 0, 1e-9, 31*86400))
def test_online_snapshot_matches_original39_at_event_and_window_boundaries(dataset, offset):
    tables = load_tables(dataset)
    clock = T0 + pd.Timedelta(seconds=offset)
    expected = build_features(tables, as_of_at=clock).set_index("customer_id")
    for uid in ("u", "v"):
        actual = observed_snapshot(tables, clock, uid)
        np.testing.assert_allclose(actual, expected.loc[uid,list(EXPANDED_FEATURES)].to_numpy(float), equal_nan=True)


@pytest.mark.parametrize("days", (27,45,90))
def test_online_snapshot_preserves_incomplete_observation_nulls(dataset, days):
    tables = load_tables(dataset)
    tables["_metadata"]["observation_start_at"] = (T0-pd.Timedelta(days=days)).isoformat()
    expected = build_features(tables, as_of_at=T0).set_index("customer_id")
    np.testing.assert_allclose(observed_snapshot(tables,T0,"u"),expected.loc["u",list(EXPANDED_FEATURES)].to_numpy(float),equal_nan=True)


def test_online_records_and_dataframe_inputs_agree_and_ignore_future_state(dataset):
    tables = load_tables(dataset)
    lists = {name: frame.to_dict("records") if isinstance(frame,pd.DataFrame) else frame for name,frame in tables.items()}
    before = observed_snapshot(lists,T0,"u")
    lists["events"].append({**lists["events"][0],"event_id":9999,"event_at":T0+pd.Timedelta(days=1),"event_type":"view"})
    np.testing.assert_allclose(before,observed_snapshot(lists,T0,"u"),equal_nan=True)


def test_online_snapshot_requires_aware_clock(dataset):
    with pytest.raises(ValueError,match="timezone-aware"):
        observed_snapshot(load_tables(dataset),"2024-03-03","u")


def test_source_entry_does_not_claim_observation_before_capture(dataset):
    tables = load_tables(dataset)
    start = T0-pd.Timedelta(days=45)
    tables['_metadata'].update(observation_start_at=start.isoformat(), time_spec='weekly_104w_v1')
    tables['customers'].loc[tables['customers'].customer_id.eq('u'), 'signed_up_at'] = start
    public = observed_snapshot(tables, T0, 'u')
    source = observed_snapshot(tables, T0, 'u', left_censored=True)
    for name in ('orders_delta_30d', 'intervisit_start_gap_mean_days_90d'):
        index = EXPANDED_FEATURES.index(name)
        assert np.isfinite(public[index]) and np.isnan(source[index])
    assert source.shape == public.shape == (39,)


@pytest.mark.parametrize('truncated,reason', [(False,'causal_policy_end'),(True,'max_events'),(True,'ongoing')])
def test_online_completion_matches_main_diagnostics_and_exact_availability(dataset, truncated, reason):
    tables = load_tables(dataset)
    tables['session_diagnostics'] = tables['sessions'][['customer_id','session_id']].assign(
        termination_reason=reason, truncated_session=truncated)
    for clock in (T0, pd.Timestamp('2024-02-20T10:40:00Z'), pd.Timestamp('2024-02-20T10:40:00Z')+pd.Timedelta(nanoseconds=1)):
        expected=build_features(tables,as_of_at=clock).set_index('customer_id')
        for uid in ('u','v'):
            np.testing.assert_allclose(observed_snapshot(tables,clock,uid),
                expected.loc[uid,list(EXPANDED_FEATURES)].to_numpy(float),equal_nan=True)
    tables['session_diagnostics']=tables['session_diagnostics'].iloc[:1]
    with pytest.raises(ValueError,match='cover every observed visit'):
        observed_snapshot(tables,T0,'u')
