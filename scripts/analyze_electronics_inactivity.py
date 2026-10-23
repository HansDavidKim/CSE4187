"""Describe observed post-second-session inactivity with a fixed follow-up window."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecommerce_sim.analysis.inactivity import inactivity_label
from ecommerce_sim.paths import CAPSTONE_ROOT


CAPSTONE = CAPSTONE_ROOT
OUT = Path('artifacts/electronics-inactivity')


def category(frame):
    return np.select([frame.purchases.gt(0), frame.carts.gt(0)],
                     ['purchase_observed', 'cart_without_purchase'], default='view_only')


def main():
    from src.irl.fit import load_events

    events, quality = load_events('electronics', CAPSTONE / 'data/raw/kaggle/electronics/events.csv')
    observation_end = events['at'].max()
    for action, name in enumerate(('views', 'carts', 'purchases')):
        events[name] = events.action.eq(action).astype('int8')
    sessions = events.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), views=('views', 'sum'),
        carts=('carts', 'sum'), purchases=('purchases', 'sum')).reset_index()
    sessions['rank'] = sessions.groupby('user', sort=False).cumcount()
    first = sessions[sessions['rank'].eq(0)].set_index('user')
    second = sessions[sessions['rank'].eq(1)].set_index('user')
    third = sessions[sessions['rank'].eq(2)].set_index('user')
    last = sessions.groupby('user', sort=False)['end'].max()
    panel = second[['start', 'end', 'views', 'carts', 'purchases']].copy()
    panel['first_type'] = category(first.reindex(panel.index))
    panel['second_type'] = category(panel)
    panel['next_start'] = third['start'].reindex(panel.index)
    panel['last_event'] = last.reindex(panel.index)
    panel['first_start'] = first['start'].reindex(panel.index)
    panel['first_end'] = first['end'].reindex(panel.index)
    panel['first_views'] = first['views'].reindex(panel.index)
    panel['first_carts'] = first['carts'].reindex(panel.index)
    panel['first_purchases'] = first['purchases'].reindex(panel.index)
    horizons = []
    for days in (14, 30, 60):
        label = inactivity_label(panel['end'], panel['next_start'], observation_end, days)
        eligible = panel[label.notna()].copy()
        eligible['inactive'] = label[label.notna()].astype(bool)
        count = len(eligible)
        inactive = eligible[eligible.inactive]
        horizons.append({'days': days, 'eligible_users': count,
                         'censored_users': int(label.isna().sum()),
                         'inactive_users': len(inactive),
                         'inactive_rate': float(len(inactive) / count) if count else None,
                         'inactive_later_return_users': int(inactive.next_start.notna().sum())})
        if days != 30:
            continue
        groups = []
        for (first_type, second_type), group in eligible.groupby(['first_type', 'second_type'], sort=True):
            groups.append({'first_type': first_type, 'second_type': second_type,
                           'users': len(group), 'inactive_users': int(group.inactive.sum()),
                           'inactive_rate': float(group.inactive.mean())})
        profiles = []
        for inactive_flag, group in eligible.groupby('inactive'):
            profiles.append({'inactive': bool(inactive_flag), 'users': len(group),
                'first_session_type_counts': group.first_type.value_counts().to_dict(),
                'second_session_type_counts': group.second_type.value_counts().to_dict(),
                'first_buyer_share': float(group.first_purchases.gt(0).mean()),
                'second_buyer_share': float(group.purchases.gt(0).mean()),
                'first_cart_share': float(group.first_carts.gt(0).mean()),
                'second_cart_share': float(group.carts.gt(0).mean()),
                'first_to_second_gap_median_days': float((group.start - group.first_end).dt.total_seconds().div(86400).median()),
                'first_events_median': float((group.first_views + group.first_carts + group.first_purchases).median()),
                'second_events_median': float((group.views + group.carts + group.purchases).median())})
        eligible['month'] = eligible['end'].dt.strftime('%Y-%m')
        month = [{'month': str(key), 'users': len(group), 'inactive_users': int(group.inactive.sum()),
                  'inactive_rate': float(group.inactive.mean())}
                 for key, group in eligible.groupby('month')]
    result = {'dataset': 'electronics', 'source_quality': quality,
              'observation_end_utc': observation_end.isoformat(), 'users_with_two_sessions': len(panel),
              'definition': 'No third observed session beginning in [end of second session, end + horizon], '
                            'with full horizon before global last logged event.',
              'horizons': horizons, 'session_type_transitions_30d': groups,
              'profiles_30d': profiles, 'landmark_month_30d': month,
              'limitations': ['Inactivity is not verified permanent churn or cause.',
                  'Sessions inferred using identifiers and 30-minute gaps; duplicate rows retained.',
                  'Observation coverage and identity linkage are not established; no activity does not prove absence.',
                  'The first two observed sessions need not be the first customer visits.',
                  'Last logged event is an administrative cutoff, not proof of continuous logging.',
                  'A session beginning exactly at the second session end is counted as observed return.',
                  'Differences are descriptive, with calendar confounding and no causal interpretation.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'horizons': horizons, 'groups_30d': groups, 'profiles_30d': profiles,
                      'landmark_month_30d': month}, indent=2))


if __name__ == '__main__':
    main()
