"""Observed-day summaries; absent dates are unknown, not no-action labels."""
from __future__ import annotations

import hashlib

import pandas as pd

COMMON = ('view', 'cart', 'purchase')


def user_split(source, user):
    digest = hashlib.sha256(f'v1:{source}:{user}'.encode()).digest()
    bucket = int.from_bytes(digest[:8], 'big') % 100
    return 'train' if bucket < 70 else 'validation' if bucket < 85 else 'test'


def build_days(events):
    """Summarize valid observed event-days; remove-containing days break trajectories."""
    required = {'user', 'at', 'action'}
    if not required.issubset(events.columns):
        raise ValueError('missing user, at or action')
    if events[['user', 'at', 'action']].isna().any().any():
        raise ValueError('input must contain valid users, timestamps and actions')
    if not events.action.isin([*COMMON, 'remove']).all():
        raise ValueError('unsupported action')
    timestamps = pd.to_datetime(events['at'], utc=True, errors='raise')
    e = events[['user', 'action']].copy()
    e['day'] = timestamps.dt.floor('D')
    flags = pd.get_dummies(e.action).reindex(columns=[*COMMON, 'remove'], fill_value=0)
    e = pd.concat([e, flags], axis=1)
    days = e.groupby(['user', 'day'], sort=True)[[*COMMON, 'remove']].max().reset_index()
    days['action_code'] = (4 * days['view'].astype(int) + 2 * days['cart'].astype(int) +
                           days['purchase'].astype(int))
    days['excluded_remove_day'] = days['remove'].astype(bool)
    days['eligible'] = ~days['excluded_remove_day'] & days['action_code'].ne(0)
    return days


def history_rows(days, max_lag=3):
    """Yield past-only states and target for consecutive eligible observed days.

    Gap is measured between previous observed eligible event-days, not between
    the state and its target; future gaps are not supplied to a model.
    """
    if max_lag not in range(4):
        raise ValueError('max_lag must be 0..3')
    result = []
    history, current_user, boundary = [], None, 'left_censored'
    for row in days.sort_values(['user', 'day'], kind='stable').itertuples(index=False):
        if row.user != current_user:
            history, current_user, boundary = [], row.user, 'left_censored'
        if not row.eligible:
            history, boundary = [], 'excluded_remove_day'
            continue
        if history:
            lag_codes = [item[0] for item in history[::-1][:max_lag + 1]]
            lag_codes += [-1] * (max_lag + 1 - len(lag_codes))
            state_gap = (history[-1][1] - history[-2][1]).days if len(history) >= 2 else None
            result.append({'user': current_user, 'target_day': row.day, 'state_day': history[-1][1],
                           'lags': tuple(lag_codes), 'state_gap_days': state_gap,
                           'history_boundary': boundary if len(history) < max_lag + 1 else 'complete',
                           'target': int(row.action_code)})
        history.append((int(row.action_code), row.day))
        history = history[-(max_lag + 1):]
    return result


def gap_bin(days):
    if days is None:
        return 'unavailable'
    if days < 1:
        raise ValueError('event-day gap must be positive')
    if days == 1:
        return '1'
    if days <= 3:
        return '2-3'
    if days <= 7:
        return '4-7'
    if days <= 30:
        return '8-30'
    return '31+'
