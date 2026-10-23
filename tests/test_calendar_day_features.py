import pandas as pd

from ecommerce_sim.calendar_day_features import calendar_days
from ecommerce_sim.observed_day_features import build_days, history_rows


def test_fill_only_internal_days_and_keep_remove_boundary():
    events = pd.DataFrame({'user': ['a'] * 4 + ['b'], 'at': pd.to_datetime([
        '2020-01-01', '2020-01-03', '2020-01-05', '2020-01-07', '2020-01-04'], utc=True),
        'action': ['view', 'purchase', 'remove', 'cart', 'view']})
    panel = calendar_days(build_days(events))
    a = panel[panel.user.eq('a')]
    assert a.action_code.tolist() == [4, 0, 1, 0, 0, 0, 2]
    assert a.assumed_inactive.tolist() == [False, True, False, True, False, True, False]
    assert a.excluded_remove_day.tolist() == [False, False, False, False, True, False, False]
    assert len(panel[panel.user.eq('b')]) == 1
    rows = history_rows(panel, 3)
    assert [r['target'] for r in rows] == [0, 1, 0, 2]
    assert rows[1]['lags'] == (0, 4, -1, -1)
    assert rows[-1]['history_boundary'] == 'excluded_remove_day'
    assert rows[-1]['lags'] == (0, -1, -1, -1)
    assert all((r['target_day'] - r['state_day']).days == 1 for r in rows)


def test_empty_input():
    empty = build_days(pd.DataFrame({'user': pd.Series(dtype='str'),
                                    'at': pd.Series(dtype='datetime64[ns, UTC]'),
                                    'action': pd.Series(dtype='str')}))
    assert calendar_days(empty).empty
