"""Cluster stability under train-user subsampling and KMeans seed changes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


ROOT = Path('/Users/kimdaewon/Desktop/캡스톤디자인')
OUT = Path('artifacts/three-ecommerce-personas')


def main(source):
    embeddings = pd.read_parquet(ROOT / f'data/models/{source}_masked_embedding_embeddings.parquet')
    members = pd.read_parquet(OUT / f'{source}_memberships.parquet')
    if not embeddings.user_id.equals(members.user_id):
        raise ValueError('Membership and embedding user order mismatch')
    z = embeddings[[f'z_{i}' for i in range(48)]].to_numpy()
    permutation = np.random.default_rng(42).permutation(len(members))
    cutoff = int(.8 * len(members))
    train, test = permutation[:cutoff], permutation[cutoff:]
    train = train[members.full_history_sessions.to_numpy()[train] >= 2]
    test = test[members.full_history_sessions.to_numpy()[test] >= 2]
    results = []
    for k in range(2, 9):
        baseline = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[train])
        reference = baseline.predict(z[test])
        if not np.array_equal(reference, members.loc[test, f'k{k}'].to_numpy()):
            raise ValueError('Reference KMeans differs from saved K sweep')
        overlap = np.zeros((5, k))
        aris = []
        for repeat in range(5):
            rng = np.random.default_rng(100 + repeat)
            sample = rng.choice(train, int(.8 * len(train)), replace=False)
            model = KMeans(n_clusters=k, n_init=10, random_state=100 + repeat).fit(z[sample])
            alternative = model.predict(z[test])
            aris.append(float(adjusted_rand_score(reference, alternative)))
            scores = np.zeros((k, k))
            for a in range(k):
                for b in range(k):
                    x, y = reference == a, alternative == b
                    union = np.logical_or(x, y).sum()
                    scores[a, b] = np.logical_and(x, y).sum() / union if union else 0
            left, right = linear_sum_assignment(-scores)
            overlap[repeat, left] = scores[left, right]
        results.append({'k': k, 'ari_replicates': aris,
                        'groups': [{'cluster': c, 'heldout_users': int((reference == c).sum()),
                                    'jaccard_replicates': overlap[:, c].tolist(),
                                    'min_jaccard': float(overlap[:, c].min()),
                                    'mean_jaccard': float(overlap[:, c].mean())} for c in range(k)]})
    (OUT / f'{source}_stability.json').write_text(json.dumps({
        'source': source, 'train_repeat_users': len(train), 'test_repeat_users': len(test),
        'method': 'five 80% train-user subsamples without replacement and five KMeans seeds; '
                  'matched held-out clusters via maximum total Jaccard',
        'limitation': 'encoder, split, sample and representation held fixed; not independent time validation',
        'results': results}, indent=2), encoding='utf-8')
    print(source, flush=True)
    for result in results:
        print('k', result['k'], 'ARI mean', round(np.mean(result['ari_replicates']), 3),
              [(g['cluster'], g['heldout_users'], round(g['mean_jaccard'], 2),
                round(g['min_jaccard'], 2)) for g in result['groups']], flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'electronics', 'retailrocket', 'cosmetics'}:
        raise SystemExit('Specify electronics, retailrocket or cosmetics')
    main(sys.argv[1])
