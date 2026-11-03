import pandas as pd

from analysis.eda import reconstruct, split_intervals


def test_overlapping_inclusive_campaigns_split_at_start_and_end_plus_one():
    campaigns = pd.DataFrame({'campaign_id': [1, 2],
        'start': pd.to_datetime(['2020-01-02', '2020-01-04']),
        'end': pd.to_datetime(['2020-01-05', '2020-01-06'])})
    p = split_intervals(campaigns, pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-07'))
    assert p.duration_days.tolist() == [1, 2, 2, 1, 1]
    assert p.campaign_ids.tolist() == [(), (1,), (1, 2), (2,), ()]
    assert p.duration_days.sum() == 7


def test_reconstruction_keeps_missing_offers_unknown_and_clips_campaign(tmp_path):
    pd.DataFrame({'campaign_id': [1, 2], 'start_date': ['02/01/20', '10/01/20'],
                  'end_date': ['06/01/20', '12/01/20']}).to_csv(tmp_path / 'campaign_data.csv', index=False)
    pd.DataFrame({'id': [1], 'campaign_id': [1], 'customer_id': [10], 'coupon_id': [3],
                  'redemption_status': [0]}).to_csv(tmp_path / 'train.csv', index=False)
    pd.DataFrame({'id': [2], 'campaign_id': [2], 'customer_id': [11], 'coupon_id': [4]}).to_csv(tmp_path / 'test.csv', index=False)
    pd.DataFrame({'date': ['2020-01-01', '2020-01-05'], 'customer_id': [10, 11],
                  'selling_price': [1, 2], 'item_id': [1, 2], 'coupon_discount': [0, 0]}).to_csv(tmp_path / 'customer_transaction_data.csv', index=False)
    periods, panel, report = reconstruct(tmp_path)
    assert len(periods) == 2
    assert report['last_period_clipped_by_observation']
    assert report['recorded_test_offer_customer_periods'] == 0
    active = panel[panel.period_id == 1].set_index('customer_id')
    assert active.loc[10, 'offer_status'] == 'recorded_offer'
    assert active.loc[11, 'offer_status'] == 'unknown_not_verified_untreated'
