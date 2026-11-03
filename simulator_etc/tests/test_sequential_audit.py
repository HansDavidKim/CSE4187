import pandas as pd
from analysis import eda as audit


def test_heartsteps_keeps_no_send_arm_and_missing_outcomes(tmp_path):
    (tmp_path / 'data_files').mkdir()
    pd.DataFrame({'user.index': [1, 1, 1], 'decision.index': [0, 1, 2],
                  'sugg.select.utime': ['2020-01-01', '2020-01-02', '2020-01-03'],
                  'avail': [True, True, False], 'connect': [True, True, False],
                  'is.randomized': [True, False, None], 'send': [True, False, None],
                  'jbsteps30pre': [1, 2, None], 'jbsteps30': [5, None, None]}).to_csv(tmp_path / 'data_files/suggestions.csv', index=False)
    data, report = audit.heartsteps(tmp_path)
    assert len(data) == 3
    assert report['available_sent'] == 1
    assert report['available_not_sent'] == 1
    assert report['available_missing_outcome'] == 1
    assert report['missing_action'] == 1


def test_journey_counts_recorded_campaigns_without_inventing_controls(tmp_path):
    pd.DataFrame({'household_key': [1, 2], 'DAY': [1, 4], 'BASKET_ID': [10, 20],
                  'SALES_VALUE': [2., 3.]}).to_csv(tmp_path / 'transaction_data.csv', index=False)
    pd.DataFrame({'household_key': [1, 1, 1], 'CAMPAIGN': [1, 1, 2]}).to_csv(tmp_path / 'campaign_table.csv', index=False)
    pd.DataFrame({'CAMPAIGN': [1, 2], 'START_DAY': [1, 3], 'END_DAY': [2, 4]}).to_csv(tmp_path / 'campaign_desc.csv', index=False)
    report = audit.journey(tmp_path)
    assert report['assignment_duplicate_keys'] == 1
    assert report['households_with_multiple_recorded_campaigns'] == 1
    assert report['assigned_households'] == 1
