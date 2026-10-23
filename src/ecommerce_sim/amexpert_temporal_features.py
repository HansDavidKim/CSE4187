"""Separate pre-decision history, offer-record proxies, behavior, and outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd


PRE = ['history_available', 'prev_purchase', 'prev_transaction_rows', 'prev_spend',
       'prev_quantity', 'prev_coupon_discount', 'prev_other_discount', 'prev_unique_items']
BEHAVIOR = ['purchase']
OUTCOME = ['transaction_rows', 'quantity', 'spend', 'other_discount', 'coupon_discount', 'unique_items']


def build_temporal_features(transactions: pd.DataFrame, campaigns: pd.DataFrame,
                            assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create a weekly panel; offer records are proxies, not verified exposure.

    Missing transaction rows are unknown, not verified non-purchases. Campaign
    dates are validity dates, not verified customer assignment timestamps.
    """
    tx = transactions.copy()
    campaigns = campaigns.copy()
    needed = ['date', 'customer_id', 'item_id', 'quantity', 'selling_price', 'other_discount', 'coupon_discount']
    if not set(needed).issubset(tx) or not {'campaign_id', 'start_date', 'end_date'}.issubset(campaigns) or not {'customer_id', 'campaign_id', 'coupon_id'}.issubset(assignments):
        raise ValueError('missing required input columns')
    if tx[needed].isna().any().any() or assignments[['customer_id', 'campaign_id', 'coupon_id']].isna().any().any():
        raise ValueError('missing required transaction or assignment values')
    tx['date'] = pd.to_datetime(tx.date, format='%Y-%m-%d', errors='raise')
    campaigns['start'] = pd.to_datetime(campaigns.start_date, format='%d/%m/%y', errors='raise')
    campaigns['end'] = pd.to_datetime(campaigns.end_date, format='%d/%m/%y', errors='raise')
    if tx.empty or campaigns[['start', 'end']].isna().any().any() or (campaigns.start > campaigns.end).any() or campaigns.campaign_id.duplicated().any():
        raise ValueError('empty transactions or invalid campaign dates/IDs')
    assignment_keys = ['customer_id', 'campaign_id', 'coupon_id']
    # Multiple rows for a coupon are not separate coupons; retain count for audit.
    duplicate_rows = int(assignments.duplicated(assignment_keys).sum())
    assign = assignments[assignment_keys].drop_duplicates().merge(
        campaigns[['campaign_id', 'start', 'end']], on='campaign_id', how='left', validate='many_to_one')
    if assign[['start', 'end']].isna().any().any():
        raise ValueError('assignment references missing campaign metadata')

    first, last = tx.date.min().normalize(), tx.date.max().normalize()
    start = first + pd.Timedelta(days=(-first.weekday()) % 7)
    end_sunday = last - pd.Timedelta(days=(last.weekday() + 1) % 7)
    end = end_sunday - pd.Timedelta(days=6)
    if end < start:
        raise ValueError('no complete observed week')
    users = np.sort(tx.customer_id.unique())
    weeks = pd.date_range(start, end, freq='7D')
    panel = pd.MultiIndex.from_product([users, weeks], names=['customer_id', 'week_start']).to_frame(index=False)
    tx['week_start'] = tx.date.dt.to_period('W-SUN').dt.start_time
    weekly = tx.groupby(['customer_id', 'week_start'], observed=True).agg(
        transaction_rows=('item_id', 'size'), quantity=('quantity', 'sum'),
        spend=('selling_price', 'sum'), other_discount=('other_discount', lambda x: -float(x.sum())),
        coupon_discount=('coupon_discount', lambda x: -float(x.sum())), unique_items=('item_id', 'nunique')).reset_index()
    panel = panel.merge(weekly, on=['customer_id', 'week_start'], how='left', validate='one_to_one')
    # shortcut: missing rows remain unknown until customer-level observation coverage is established.
    panel['purchase'] = pd.Series(pd.NA, index=panel.index, dtype='Int8')
    panel.loc[panel.transaction_rows.notna(), 'purchase'] = 1
    panel = panel.sort_values(['customer_id', 'week_start']).reset_index(drop=True)
    group = panel.groupby('customer_id', sort=False)
    panel['history_available'] = group.cumcount() > 0
    for col in ['purchase', *OUTCOME]:
        panel['prev_' + col] = group[col].shift(1)

    panel['week_end_exclusive'] = panel.week_start + pd.Timedelta(days=7)
    report = {'rows': len(panel), 'customers': len(users), 'weeks': len(weeks),
        'first_week': str(start.date()), 'last_week': str(end.date()),
        'roles': {'identifier_and_time': ['customer_id', 'week_start', 'week_end_exclusive'],
                  'predecision_history': PRE, 'intervention': [],
                  'behavior': BEHAVIOR, 'postdecision_outcomes': OUTCOME},
        'purchase_unknown_rows': int(panel.purchase.isna().sum()),
        'assignment_distinct_customer_campaign_coupon_records_audit_only': len(assign),
        'first_week_history_unavailable_rows': int((~panel.history_available).sum()),
        'assignment_duplicate_customer_campaign_coupon_extra_rows': duplicate_rows,
        'excluded_transaction_rows_partial_weeks': int(((tx.week_start < start) | (tx.week_start > end)).sum()),
        'caveats': ['Offer assignment and delivery time are unknown; no offer record is joined to customer-week rows.',
            'Campaign validity dates do not establish individual assignment or exposure time.',
            'Missing transaction rows are unknown, not confirmed non-purchases.',
            'Customer inclusion is based on transactions over the full source period; this panel is not a real-time eligible population.',
            'Current and lagged variables overlap deterministically; select nonredundant nodes and lags before causal discovery.',
            'Redemption_status is excluded: its exact redemption timing is unavailable.',
            'Feature roles and lags are bookkeeping, not proof of causal graph edges.',
            'Keep individual-level derived data local pending rights review.']}
    return panel[['customer_id', 'week_start', 'week_end_exclusive', *PRE, *BEHAVIOR, *OUTCOME]], report
