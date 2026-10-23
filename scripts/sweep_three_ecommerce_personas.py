"""Sweep K on three existing dataset-specific masked-GRU representations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.analysis.masked_embedding_clusters import rows_from_events, source_data


CAPSTONE = Path('/Users/kimdaewon/Desktop/캡스톤디자인')
OUT = Path('artifacts/three-ecommerce-personas')


def run(source: str):
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    embeddings = pd.read_parquet(CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if embeddings.user_id.tolist() != [row[0] for row in rows]:
        raise ValueError(f'{source}: reconstructed sample differs from saved embeddings')
    z = embeddings[[f'z_{i}' for i in range(48)]].to_numpy()
    sessions = np.array([row[2] for row in rows])
    permutation = np.random.default_rng(42).permutation(len(rows))
    train, test = permutation[:int(.8 * len(rows))], permutation[int(.8 * len(rows)):]
    split = np.full(len(rows), 'train', dtype=object)
    split[test] = 'test'
    train = train[sessions[train] >= 2]
    test = test[sessions[test] >= 2]
    profiles = []
    for i in test:
        tokens = rows[i][1]
        record = {name: tokens.count(j) for j, name in enumerate(actions)}
        record['tokens'] = len(tokens)
        for origin, dest in [('view', 'cart'), ('cart', 'purchase'), ('cart', 'remove')]:
            if dest not in actions:
                continue
            a, b = actions.index(origin), actions.index(dest)
            pairs = list(zip(tokens, tokens[1:]))
            record[origin + '_next_count'] = sum(x == a for x, _ in pairs)
            record[origin + '_to_' + dest] = sum(x == a and y == b for x, y in pairs)
        profiles.append(record)
    profiles = pd.DataFrame(profiles)
    members = pd.DataFrame({'user_id': embeddings.user_id, 'split': split,
                            'full_history_sessions': sessions})
    sweep = []
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
        alt = KMeans(n_clusters=k, n_init=10, random_state=43).fit(z[train])
        labels = model.predict(z[test])
        members[f'k{k}'] = model.predict(z)
        groups = []
        for c in range(k):
            frame = profiles[labels == c]
            counts = frame[actions].sum()
            group = {'cluster': c, 'train_users': int((model.labels_ == c).sum()),
                     'heldout_users': len(frame),
                     'event_shares': (counts / counts.sum()).to_dict() if counts.sum() else None,
                     'user_action_shares': {a: float(frame[a].gt(0).mean()) if len(frame) else None
                                            for a in actions},
                     'median_tokens': float(frame.tokens.median()) if len(frame) else None}
            for origin, dest in [('view', 'cart'), ('cart', 'purchase'), ('cart', 'remove')]:
                if dest not in actions:
                    continue
                n = int(frame[origin + '_next_count'].sum())
                group[origin + '_to_' + dest] = {'rate': float(frame[origin + '_to_' + dest].sum() / n)
                                                  if n else None, 'denominator': n}
            groups.append(group)
        assert sum(g['heldout_users'] for g in groups) == len(test)
        sweep.append({'k': k, 'kmeans_seed_ari': float(adjusted_rand_score(labels, alt.predict(z[test]))),
                      'clusters': groups})
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'source': source, 'quality': quality, 'actions': actions, 'users': len(rows),
              'mature_train_users': len(train), 'mature_test_users': len(test),
              'sweep': sweep}
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    members.to_parquet(OUT / f'{source}_memberships.parquet', index=False)
    print(f'{source}: users={len(rows)}, train/test repeat={len(train)}/{len(test)}', flush=True)
    for item in sweep:
        print('k=', item['k'], 'test sizes=', [g['heldout_users'] for g in item['clusters']], flush=True)


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:] or ['electronics', 'retailrocket', 'cosmetics']:
        if name not in {'electronics', 'retailrocket', 'cosmetics'}:
            raise ValueError(name)
        run(name)
