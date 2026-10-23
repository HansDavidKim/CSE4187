"""Anchor users at a common date and examine subsequent 60-day inactivity."""
import json
import sys

import pandas as pd

from scripts.audit_observed_days import ROOT, load_events


def summarize(events, first_complete, end_exclusive):
    anchor = end_exclusive - pd.Timedelta(days=60)
    if anchor <= first_complete + pd.Timedelta(days=30):
        raise ValueError('need 30 days of history and 60 days of follow-up')
    e = events[(events['at'] >= first_complete) & (events['at'] < end_exclusive)].copy()
    e['day'] = e['at'].dt.floor('D')
    past = e[e['day'] < anchor].copy()
    future = e[e['day'] >= anchor]
    days = past[['user', 'day']].drop_duplicates().sort_values(['user', 'day'])
    users = days.groupby('user', sort=False)['day'].agg(['min', 'max', 'size'])
    users['days_since_visit'] = (anchor - users['max']).dt.days
    for period in (7, 30):
        cnt = days[days['day'] >= anchor - pd.Timedelta(days=period)].groupby('user').size()
        users[f'visits_{period}'] = cnt.reindex(users.index, fill_value=0)
    future_day = future[['user', 'day']].drop_duplicates()
    returned_30 = set(future_day.loc[future_day.day < anchor + pd.Timedelta(days=30), 'user'])
    returned_60 = set(future_day.user)
    users['status'] = 'inactive60'
    users.loc[users.index.isin(returned_60), 'status'] = 'late_return_31_60'
    users.loc[users.index.isin(returned_30), 'status'] = 'return_1_30'
    last = days.groupby('user', sort=False).tail(1)
    last_events = past.merge(last, on=['user', 'day'], how='inner')
    flags = pd.crosstab(last_events['user'], last_events['action']).gt(0)
    users = users.join(flags, how='left')
    recent = days.groupby('user', sort=False).tail(4).copy()
    recent['gap'] = recent.groupby('user', sort=False)['day'].diff().dt.days
    recent['rank'] = recent.groupby('user', sort=False).cumcount()
    gaps = recent[recent['rank'] > 0].pivot(index='user', columns='rank', values='gap').reindex(columns=[1, 2, 3]).dropna()
    # Only users with four pre-anchor days have three valid within-history gaps.
    assert all(users.loc[gaps.index, 'size'] >= 4)
    def group_summary(u):
        g = gaps.loc[gaps.index.intersection(u.index)]
        n = len(u)
        return {'users': n,
            'past_single_visit_day': int(u['size'].eq(1).sum()),
            'days_since_visit_median': float(u['days_since_visit'].median()) if n else None,
            'visit_days_last_7_mean': float(u['visits_7'].mean()) if n else None,
            'visit_days_last_30_mean': float(u['visits_30'].mean()) if n else None,
            'last_visit_action_fraction': {a: float(u[a].fillna(False).mean()) if a in u and n else 0.0
                                           for a in ('view', 'cart', 'purchase', 'remove')},
            'four_or_more_pre_anchor_visit_days': len(g),
            'last_three_gaps_median': [float(g[i].median()) for i in (1, 2, 3)] if len(g) else None,
            'gap_increasing_fraction': float(((g[1] < g[2]) & (g[2] < g[3])).mean()) if len(g) else None,
            'third_gap_greater_first_fraction': float((g[3] > g[1]).mean()) if len(g) else None,
            'third_gap_less_first_fraction': float((g[3] < g[1]).mean()) if len(g) else None}
    groups, recent_groups = {}, {}
    for label in ('inactive60', 'late_return_31_60', 'return_1_30'):
        u = users[users.status.eq(label)]
        groups[label] = group_summary(u)
        recent_groups[label] = group_summary(u[u['visits_30'] > 0])
    return {'anchor_utc': anchor.isoformat(), 'end_exclusive_utc': end_exclusive.isoformat(),
            'start_complete_utc': first_complete.isoformat(), 'groups': groups,
            'groups_with_pre_anchor_visit_in_last_30_days': recent_groups}


def run(source):
    events, loader = load_events(source)
    result = summarize(events, events['at'].min().ceil('D'), events['at'].max().floor('D'))
    result.update({'source': source, 'loader': loader,
                   'cosmetics_sampling': 'user id modulo 100 == 0' if source == 'cosmetics' else None,
                   'definition': 'Anchor is start of final 60 full UTC days. Cohort has at least one prior logged event; groups are mutually exclusive by first return within 60 days.',
                   'caveats': ['60-day inactivity is not verified permanent churn.',
                     'All features use only pre-anchor events; last visit means latest pre-anchor visit, not necessarily final visit.',
                     'Single fixed anchor; observation length and user-entry timing differ; not matched or causal.',
                     'Some users already had long inactivity before anchor; compare recency and consider first-seen cohorts.',
                     'Only completed observed pre-anchor gaps are used; requiring >=4 days selects repeat visitors.',
                     'Platform logging/identity completeness is assumed; cosmetics is a modulo sample.']})
    out = ROOT / 'artifacts/long-inactivity-analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps(result), flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
