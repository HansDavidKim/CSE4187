"""Reconstruct coupon-validity intervals without inventing untreated labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def split_intervals(campaigns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Split inclusive observation window at campaign starts and day after ends."""
    if start > end or campaigns[['start', 'end']].isna().any().any() or (campaigns.start > campaigns.end).any():
        raise ValueError('invalid campaign or observation dates')
    stop = end + pd.Timedelta(days=1)
    cuts = {start, stop}
    for row in campaigns.itertuples():
        if row.start <= end and row.end >= start:
            cuts.add(max(row.start, start))
            cuts.add(min(row.end + pd.Timedelta(days=1), stop))
    bounds = sorted(cuts)
    rows = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        active = campaigns.loc[(campaigns.start <= left) & (campaigns.end >= left), 'campaign_id']
        rows.append({'period_id': len(rows), 'start': left, 'end_exclusive': right,
                     'duration_days': (right-left).days, 'campaign_ids': tuple(sorted(active.tolist()))})
    return pd.DataFrame(rows)


def reconstruct(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    campaign = pd.read_csv(root / 'campaign_data.csv')
    campaign['start'] = pd.to_datetime(campaign.start_date, format='%d/%m/%y')
    campaign['end'] = pd.to_datetime(campaign.end_date, format='%d/%m/%y')
    train = pd.read_csv(root / 'train.csv')
    test = pd.read_csv(root / 'test.csv')
    tx = pd.read_csv(root / 'customer_transaction_data.csv')
    tx['date'] = pd.to_datetime(tx.date, format='%Y-%m-%d')
    if campaign.campaign_id.duplicated().any() or train.id.duplicated().any() or test.id.duplicated().any():
        raise ValueError('duplicate campaign or assignment ID')
    if set(train.id) & set(test.id):
        raise ValueError('train/test assignment IDs overlap')
    if not set(train.campaign_id).union(test.campaign_id).issubset(set(campaign.campaign_id)):
        raise ValueError('assignment campaign missing from campaign metadata')
    train_ids, test_ids = set(train.campaign_id), set(test.campaign_id)
    intervals = split_intervals(campaign, tx.date.min(), tx.date.max())
    customers = np.union1d(tx.customer_id.unique(), np.union1d(train.customer_id.unique(), test.customer_id.unique()))
    interval_count = len(intervals)
    lookup = pd.DataFrame({'customer_id': customers})
    panel = lookup.merge(intervals.drop(columns='campaign_ids'), how='cross')
    bounds = pd.DatetimeIndex(intervals.start).to_numpy()
    period_index = np.searchsorted(bounds, tx.date.to_numpy(), side='right') - 1
    if (period_index < 0).any() or (period_index >= interval_count).any():
        raise ValueError('transaction outside reconstructed intervals')
    tx['period_id'] = period_index
    spend = tx.groupby(['customer_id', 'period_id']).agg(spend=('selling_price', 'sum'),
        transaction_rows=('item_id', 'size'), coupon_discount=('coupon_discount', 'sum')).reset_index()
    panel = panel.merge(spend, on=['customer_id', 'period_id'], how='left', validate='one_to_one')
    for col in ('spend', 'transaction_rows', 'coupon_discount'):
        panel[col] = panel[col].fillna(0)
    panel['spend_per_day'] = panel.spend / panel.duration_days
    panel['has_train_offer_record'] = False
    panel['has_test_offer_record'] = False
    for source, frame in [('train', train), ('test', test)]:
        counts = frame.groupby(['customer_id', 'campaign_id']).size().rename('count').reset_index()
        occurrences = []
        for period in intervals.itertuples():
            if period.campaign_ids:
                segment = counts[counts.campaign_id.isin(period.campaign_ids)][['customer_id']].drop_duplicates().copy()
                segment['period_id'] = period.period_id
                occurrences.append(segment)
        if occurrences:
            idx = pd.MultiIndex.from_frame(pd.concat(occurrences).drop_duplicates())
            panel[f'has_{source}_offer_record'] = pd.MultiIndex.from_frame(panel[['customer_id','period_id']]).isin(idx)
    panel['offer_status'] = np.where(panel.has_train_offer_record | panel.has_test_offer_record,
                                     'recorded_offer', 'unknown_not_verified_untreated')
    panel = panel.sort_values(['customer_id', 'period_id']).reset_index(drop=True)
    panel['previous_spend_per_day'] = panel.groupby('customer_id').spend_per_day.shift(1)
    panel['previous_transaction_rows_per_day'] = (panel.transaction_rows / panel.duration_days).groupby(panel.customer_id).shift(1)
    panel['previous_has_offer_record'] = panel.groupby('customer_id').has_train_offer_record.shift(1)
    panel['previous_period_days'] = panel.groupby('customer_id').duration_days.shift(1)
    exposure = train[['campaign_id','customer_id','coupon_id','redemption_status']]
    test_overlap = set(test.campaign_id) & train_ids
    unknown_ids = set(campaign.campaign_id) - train_ids - test_ids
    report = {'observation_start': str(tx.date.min().date()), 'observation_end': str(tx.date.max().date()),
        'customers': len(customers), 'periods': interval_count, 'active_periods': int(intervals.campaign_ids.map(bool).sum()),
        'rows': len(panel), 'duration_days_min': int(intervals.duration_days.min()),
        'duration_days_max': int(intervals.duration_days.max()),
        'campaigns_total': len(campaign), 'train_campaigns': sorted(map(int, train_ids)),
        'test_campaigns': sorted(map(int, test_ids)), 'train_test_campaign_overlap': sorted(map(int, test_overlap)),
        'campaigns_without_assignment_records': sorted(map(int, unknown_ids)),
        'assignment_ids_train': len(train), 'assignment_ids_test': len(test),
        'train_duplicate_customer_campaign_coupon_rows': int(exposure.duplicated(['customer_id','campaign_id','coupon_id']).sum()),
        'test_duplicate_customer_campaign_coupon_rows': int(test.duplicated(['customer_id','campaign_id','coupon_id']).sum()),
        'recorded_train_offer_customer_periods': int(panel.has_train_offer_record.sum()),
        'recorded_test_offer_customer_periods': int(panel.has_test_offer_record.sum()),
        'recorded_either_offer_customer_periods': int((panel.offer_status == 'recorded_offer').sum()),
        'unknown_treatment_customer_periods': int((panel.offer_status != 'recorded_offer').sum()),
        'redemption_positive_train_rows': int(train.redemption_status.sum()),
        'analysis_rows_after_initial_history_period': int((panel.period_id > 0).sum()),
        'recorded_offer_share_after_initial_period': float(panel.loc[panel.period_id > 0, 'has_train_offer_record'].mean()),
        'last_period_clipped_by_observation': bool(((campaign.campaign_id.isin(intervals.iloc[-1].campaign_ids)) & (campaign.end > tx.date.max())).any()),
        'test_redemption_labels_available': False,
        'caveats': ['Observed test offers are used only as recorded assignments; test redemption labels are absent.',
            'No assignment record is not verified as no exposure; do not interpret unknown as an untreated control.',
            'Interval cut points use campaign metadata, not verified actual assignment timestamps.',
            'Observation boundaries clip campaigns; incomplete edge intervals are not full treatment windows.',
            'No randomized assignment or conditional exchangeability has been established.',
            'Raw and row-level derivatives remain local pending a data rights review.']}
    return intervals, panel, report
