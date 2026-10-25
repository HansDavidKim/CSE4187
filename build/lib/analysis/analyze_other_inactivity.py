"""Evaluate 30-day inactivity in Retailrocket and sampled Cosmetics logs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.inactivity import inactivity_label
from utils.paths import CAPSTONE_ROOT


CAPSTONE = CAPSTONE_ROOT
OUT = Path('artifacts/other-inactivity')


def category(frame):
    return np.select([frame.purchases.gt(0), (frame.carts + frame.removes).gt(0)],
                     ['purchase_observed', 'cart_or_remove_no_purchase'], default='view_only')


def run(source):
    from analysis.masked_embedding_clusters import source_data

    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else 'data/raw/kaggle/retailrocket/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    first_time, end_time = events['at'].min(), events['at'].max()
    daily = events['at'].dt.floor('D').value_counts().sort_index()
    missing_days = pd.date_range(first_time.floor('D'), end_time.floor('D'), freq='D').difference(daily.index)
    for name in ('view', 'cart', 'purchase', 'remove'):
        events[name + 's'] = (events.action.eq(actions.index(name)).astype('int8')
                              if name in actions else 0)
    sessions = events.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), views=('views', 'sum'),
        carts=('carts', 'sum'), purchases=('purchases', 'sum'), removes=('removes', 'sum')).reset_index()
    sessions['rank'] = sessions.groupby('user', sort=False).cumcount()
    first = sessions[sessions['rank'].eq(0)].set_index('user')
    second = sessions[sessions['rank'].eq(1)].set_index('user')
    third = sessions[sessions['rank'].eq(2)].set_index('user')
    panel = second[['start', 'end', 'views', 'carts', 'purchases', 'removes']].copy()
    panel['first_type'] = category(first.reindex(panel.index))
    panel['second_type'] = category(panel)
    panel['next_start'] = third['start'].reindex(panel.index)
    panel['first_views'] = first['views'].reindex(panel.index)
    panel['first_carts'] = first['carts'].reindex(panel.index)
    panel['first_purchases'] = first['purchases'].reindex(panel.index)
    panel['first_removes'] = first['removes'].reindex(panel.index)
    horizons = []
    for days in (14, 30, 60):
        labels = inactivity_label(panel.end, panel.next_start, end_time, days)
        eligible = panel[labels.notna()].copy()
        eligible['inactive'] = labels[labels.notna()].astype(bool)
        inactive = eligible[eligible.inactive]
        horizons.append({'days': days, 'eligible_users': len(eligible),
                         'censored_users': int(labels.isna().sum()), 'inactive_users': len(inactive),
                         'inactive_rate': float(inactive.shape[0] / len(eligible)),
                         'inactive_later_return_users': int(inactive.next_start.notna().sum())})
        if days != 30:
            continue
        groups = [{'first_type': a, 'second_type': b, 'users': len(group),
                   'inactive_users': int(group.inactive.sum()), 'inactive_rate': float(group.inactive.mean())}
                  for (a, b), group in eligible.groupby(['first_type', 'second_type'], sort=True)]
        profiles = [{'inactive': bool(label), 'users': len(group),
                     'second_session_type_counts': group.second_type.value_counts().to_dict(),
                     'second_buyer_share': float(group.purchases.gt(0).mean()),
                     'second_cart_share': float(group.carts.gt(0).mean()),
                     'second_remove_share': (float(group.removes.gt(0).mean()) if 'remove' in actions else None),
                     'first_events_median': float((group.first_views + group.first_carts + group.first_purchases + group.first_removes).median()),
                     'second_events_median': float((group.views + group.carts + group.purchases + group.removes).median())}
                    for label, group in eligible.groupby('inactive')]
        eligible['month'] = eligible['end'].dt.strftime('%Y-%m')
        monthly = [{'month': str(month), 'users': len(group), 'inactive_users': int(group.inactive.sum()),
                    'inactive_rate': float(group.inactive.mean())}
                   for month, group in eligible.groupby('month')]
    result = {'source': source, 'quality': quality, 'observation_start_utc': first_time.isoformat(),
              'observation_end_utc': end_time.isoformat(), 'observed_daily_dates': len(daily),
              'zero_event_dates': [x.date().isoformat() for x in missing_days],
              'users_with_two_sessions': len(panel), 'horizons': horizons,
              'session_type_transitions_30d': groups, 'profiles_30d': profiles,
              'landmark_month_30d': monthly,
              'limitations': ['30-day absence is not verified churn or a causal outcome.',
                              'Observed sessions and IDs do not guarantee complete behavior coverage.',
                              'First observed sessions need not be first visits.',
                              'Observation cutoff is last recorded event, not verified logging coverage.',
                              'Dataset-specific sessionization and sampled populations prevent direct comparisons.',
                              'Cosmetics uses user_id modulo 100 and excludes missing user/session rows.'
                              if source == 'cosmetics' else 'Retailrocket has no removal event; it is unobserved.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'retailrocket', 'cosmetics'}:
        raise SystemExit('Specify retailrocket or cosmetics')
    run(sys.argv[1])
