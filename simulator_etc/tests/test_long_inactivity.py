import pandas as pd
from analysis.eda import summarize_long_inactivity


def test_anchor_groups_and_no_future_leakage():
    events = pd.DataFrame({'user': ['a', 'a', 'b', 'b', 'c', 'c'],
        'at': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-02-01',
                              '2020-03-10', '2020-02-01', '2020-04-05'], utc=True),
        'action': ['view', 'purchase', 'view', 'purchase', 'view', 'cart']})
    result = summarize_long_inactivity(events, pd.Timestamp('2020-01-01', tz='UTC'), pd.Timestamp('2020-05-01', tz='UTC'))
    groups = result['groups']
    assert groups['inactive60']['users'] == 1
    assert groups['return_1_30']['users'] == 1
    assert groups['late_return_31_60']['users'] == 1
    assert groups['return_1_30']['last_visit_action_fraction']['purchase'] == 0
    assert groups['inactive60']['last_visit_action_fraction']['purchase'] == 1
