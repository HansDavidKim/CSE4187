import pandas as pd
from scripts.analyze_visit_gaps import analyze


def test_first_three_gaps_and_short_users():
    events = pd.DataFrame({'user': ['a'] * 5 + ['b'], 'at': pd.to_datetime([
        '2020-01-01', '2020-01-02', '2020-01-04', '2020-01-07', '2020-01-07',
        '2020-01-01'], utc=True)})
    r = analyze(events)
    assert r['users_total'] == 2 and r['users_with_4plus_visit_days'] == 1
    assert r['patterns']['strictly_increasing']['users'] == 1
    assert r['strict_increase_examples'][0]['gaps_days'] == [1, 2, 3]
