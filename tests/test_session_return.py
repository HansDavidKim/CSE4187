import pandas as pd
from ecommerce_sim.session_return import durations, expand_hazards, sessions


def test_session_gap_and_censor_at_training_cutoff():
    e = pd.DataFrame({'user': ['u'] * 3, 'at': pd.to_datetime([
        '2020-01-01 00:00', '2020-01-01 00:10', '2020-01-02 00:00'], utc=True),
        'action': ['view', 'cart', 'purchase']})
    s = sessions(e, pd.Timestamp('2020-02-01', tz='UTC'))
    assert len(s) == 2 and s.iloc[0]['end'] == pd.Timestamp('2020-01-01 00:40', tz='UTC')
    elapsed, event, _ = durations(s.iloc[:1], pd.Timestamp('2020-01-01 12:40', tz='UTC'))
    assert elapsed[0] == 12 and not event[0]
    x, y = expand_hazards(s.iloc[:1], pd.Timestamp('2020-01-01 12:40', tz='UTC'))
    assert len(x) == 2 and not y.any()
    assert s.iloc[1]['has_previous'] == 1


def test_early_return_not_included_if_interval_followup_incomplete():
    e = pd.DataFrame({'user': ['u', 'u'], 'at': pd.to_datetime([
        '2020-01-01 00:00', '2020-01-01 02:00'], utc=True), 'action': ['view', 'view']})
    s = sessions(e, pd.Timestamp('2020-01-03', tz='UTC'))
    x, y = expand_hazards(s.iloc[:1], pd.Timestamp('2020-01-01 03:00', tz='UTC'))
    assert len(x) == 1 and not y.any()
