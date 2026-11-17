import pandas as pd
import pytest

from preprocessing.amexpert_temporal_features import PRE, build_temporal_features


def fixture():
    tx = pd.DataFrame([
        ['2020-01-06', 1, 1, 1, 10., -1., 0.],
        ['2020-01-13', 1, 2, 2, 20., 0., -2.],
        ['2020-01-20', 2, 3, 1, 5., 0., 0.],
        ['2020-01-26', 1, 4, 1, 6., 0., 0.],
    ], columns=['date', 'customer_id', 'item_id', 'quantity', 'selling_price', 'other_discount', 'coupon_discount'])
    campaigns = pd.DataFrame([[1, '07/01/20', '19/01/20'], [2, '20/01/20', '26/01/20']],
                             columns=['campaign_id', 'start_date', 'end_date'])
    assignments = pd.DataFrame([[1, 1, 10], [1, 1, 10], [2, 2, 11]],
                               columns=['customer_id', 'campaign_id', 'coupon_id'])
    return tx, campaigns, assignments


def test_roles_lags_unknown_exposure_and_temporal_boundaries():
    tx, campaigns, assignments = fixture()
    panel, report = build_temporal_features(tx, campaigns, assignments)
    one = panel[panel.customer_id == 1].reset_index(drop=True)
    assert one.purchase.tolist() == [1, 1, 1]
    assert pd.isna(one.loc[0, 'prev_purchase'])
    assert one.prev_purchase.iloc[1:].tolist() == [1, 1]
    assert one.prev_spend.iloc[1:].tolist() == [10, 20]
    assert not any('offer' in col or 'campaign' in col for col in panel.columns)
    assert report['roles']['intervention'] == []
    assert report['assignment_duplicate_customer_campaign_coupon_extra_rows'] == 1
    assert set(report['roles']['predecision_history']) == set(PRE)
    assert not set(report['roles']['postdecision_outcomes']) & set(report['roles']['predecision_history'])
    two = panel[panel.customer_id == 2].reset_index(drop=True)
    assert two.purchase.isna().tolist() == [True, True, False]
    assert two.spend.isna().tolist() == [True, True, False]


def test_late_assignment_record_cannot_change_weekly_panel():
    tx, campaigns, assignments = fixture()
    original, _ = build_temporal_features(tx, campaigns, assignments)
    updated, _ = build_temporal_features(tx, campaigns, assignments.iloc[0:0])
    pd.testing.assert_frame_equal(original, updated)


def test_current_outcome_and_future_campaign_do_not_change_prior_history():
    tx, campaigns, assignments = fixture()
    original, _ = build_temporal_features(tx, campaigns, assignments)
    tx.loc[tx.date == '2020-01-13', 'selling_price'] = 999.
    updated, _ = build_temporal_features(tx, campaigns, assignments)
    key = (updated.customer_id == 1) & (updated.week_start == pd.Timestamp('2020-01-13'))
    pd.testing.assert_frame_equal(original.loc[key, PRE], updated.loc[key, PRE])
    assert original.loc[key, 'spend'].iloc[0] != updated.loc[key, 'spend'].iloc[0]


def test_reject_missing_campaign_id():
    tx, campaigns, assignments = fixture()
    assignments.loc[0, 'campaign_id'] = 999
    with pytest.raises(ValueError, match='missing campaign'):
        build_temporal_features(tx, campaigns, assignments)
