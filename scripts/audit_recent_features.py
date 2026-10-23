"""Audit recent-state coverage, without training IRL or storing user records."""
import json
import sys

import pandas as pd

from ecommerce_sim.preprocessing.recent_history_features import ACTIONS, decision_rows, feature_names
from scripts.run_history_trajectory_irl import CAPSTONE, ROOT


def run(source):
    from src.analysis.masked_embedding_clusters import source_data

    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, _ = source_data(source, path, sample_modulus=100)
    members = pd.read_parquet(ROOT / 'artifacts/three-ecommerce-personas' / f'{source}_memberships.parquet')
    if members.user_id.duplicated().any():
        raise ValueError('duplicate user assignments')
    selected = events[events.user.isin(members.user_id)].groupby('user', sort=False).head(64)
    split = members.set_index('user_id')['split'].to_dict()
    summaries = {k: {'users': 0, 'decisions': 0, 'action_counts': [0]*3, 'available_lags': [0]*4,
                     'tied_events_skipped': 0, 'unsupported_events': 0} for k in ('train', 'test')}
    for user, frame in selected.groupby('user', sort=False):
        rows, audit = decision_rows(frame, source)
        group = summaries[split[user]]
        group['users'] += bool(rows)
        group['decisions'] += len(rows)
        for key, value in audit.items():
            group[key] += value
        for recent, boundary, action in rows:
            group['action_counts'][action] += 1
            for lag in range(4):
                group['available_lags'][lag] += len(recent) > lag
    result = {'source': source, 'source_quality': quality, 'actions': ACTIONS,
              'features_by_max_lag': {str(lag): feature_names(lag) for lag in range(4)},
              'splits': summaries,
              'definition': 'lag0 is latest previous action; lag k is k actions older; target is the next observed action.',
              'tie_rule': 'skip all same-user same-session same-timestamp events and reset recent context',
              'unsupported_rule': 'exclude the unmodeled event and reset recent context',
              'sample_rule': 'existing retrospective stratified user cohort; first 64 events per user',
              'no_model_fitted': True}
    out = ROOT / 'artifacts/recent-feature-audit'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, summaries, flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
