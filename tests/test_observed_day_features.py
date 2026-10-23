import pandas as pd

from ecommerce_sim.observed_day_features import build_days, gap_bin, history_rows, user_split


def test_ties_duplicates_and_remove_are_not_zero_actions():
    e = pd.DataFrame({'user': [1]*7, 'at': pd.to_datetime([
        '2020-01-01', '2020-01-01', '2020-01-01', '2020-01-03',
        '2020-01-04', '2020-01-05', '2020-01-07'], utc=True),
        'action': ['view', 'cart', 'view', 'remove', 'purchase', 'view', 'cart']})
    a = build_days(e)
    b = build_days(e.iloc[::-1])
    pd.testing.assert_frame_equal(a, b)
    assert a.action_code.tolist() == [6, 0, 1, 4, 2]
    assert not a.iloc[1].eligible
    rows = history_rows(a, 3)
    assert [r['target'] for r in rows] == [4, 2]
    assert rows[0]['lags'] == (1, -1, -1, -1)
    assert rows[0]['history_boundary'] == 'excluded_remove_day'
    assert rows[1]['lags'] == (4, 1, -1, -1)
    assert rows[1]['state_gap_days'] == 1
    assert all(r['state_day'] < r['target_day'] for r in rows)


def test_initial_history_is_left_censored_not_no_action():
    e = pd.DataFrame({'user': ['u', 'u'], 'at': pd.to_datetime(
        ['2020-01-01', '2020-01-05'], utc=True), 'action': ['view', 'purchase']})
    row = history_rows(build_days(e), 3)[0]
    assert row['history_boundary'] == 'left_censored'
    assert row['lags'] == (4, -1, -1, -1)
    assert row['state_gap_days'] is None


def test_split_and_gap():
    assert user_split('electronics', 123) == user_split('electronics', 123)
    assert [gap_bin(x) for x in [None, 1, 2, 4, 8, 31]] == [
        'unavailable', '1', '2-3', '4-7', '8-30', '31+']
