"""Compare observed third sessions by 30-day subsequent inactivity."""
import json
import sys
from pathlib import Path

import numpy as np
from ecommerce_sim.analysis.inactivity import inactivity_label
from ecommerce_sim.paths import CAPSTONE_ROOT

ROOT = CAPSTONE_ROOT
OUT = Path('artifacts/late-inactivity')


def session_type(frame):
    return np.select([frame.purchases.gt(0), (frame.carts + frame.removes).gt(0)],
                     ['purchase', 'cart_or_remove_no_purchase'], default='view_only')


def run(source):
    from src.analysis.masked_embedding_clusters import source_data

    path = ROOT / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                   else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    end = events['at'].max()
    for name in ('view', 'cart', 'remove', 'purchase'):
        events[name + 's'] = (events.action.eq(actions.index(name)).astype('int8')
                              if name in actions else 0)
    sessions = events.groupby(['user', 'session'], sort=False).agg(
        start=('at', 'min'), end=('at', 'max'), views=('views', 'sum'),
        carts=('carts', 'sum'), removes=('removes', 'sum'), purchases=('purchases', 'sum')).reset_index()
    sessions = sessions.sort_values(['user', 'start', 'end', 'session'], kind='stable')
    sessions['rank'] = sessions.groupby('user', sort=False).cumcount()
    second = sessions[sessions['rank'].eq(1)].set_index('user')
    third = sessions[sessions['rank'].eq(2)].set_index('user').copy()
    fourth = sessions[sessions['rank'].eq(3)].set_index('user')
    third['previous_type'] = session_type(second.reindex(third.index))
    third['type'] = session_type(third)
    third['next_start'] = fourth['start'].reindex(third.index)
    labels = inactivity_label(third.end, third.next_start, end, 30)
    panel = third[labels.notna()].copy()
    panel['inactive'] = labels[labels.notna()].astype(bool)
    profiles = []
    for label, group in panel.groupby('inactive'):
        transitions = group.groupby(['previous_type', 'type']).size().to_dict()
        profiles.append({'inactive_30d': bool(label), 'users': len(group),
                         'third_session_types': group.type.value_counts().to_dict(),
                         'third_view_any': int(group.views.gt(0).sum()),
                         'third_cart_any': int(group.carts.gt(0).sum()),
                         'third_purchase_any': int(group.purchases.gt(0).sum()),
                         'third_remove_any': int(group.removes.gt(0).sum()) if 'remove' in actions else None,
                         'third_events_median': float(group[['views', 'carts', 'removes', 'purchases']].sum(axis=1).median()),
                         'second_to_third': [{'second': a, 'third': b, 'users': int(n)}
                                             for (a,b), n in transitions.items()]})
    result = {'source': source, 'quality': quality, 'observation_end': end.isoformat(),
              'at_least_three_sessions': len(third), 'eligible': len(panel),
              'censored': int(labels.isna().sum()), 'profiles': profiles,
              'limitations': ['Third observed session is not necessarily the final lifetime session.',
                              'No observed event is not proof of churn or complete logging.',
                              '30-day eligibility uses last observed timestamp as cutoff.',
                              'Cosmetics uses deterministic user_id modulo 100 sampling.' if source == 'cosmetics' else
                              'Electronics has no remove event in the input action set.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    for dataset in sys.argv[1:]:
        if dataset not in ('electronics', 'cosmetics'):
            raise SystemExit('Choose electronics or cosmetics')
        run(dataset)
