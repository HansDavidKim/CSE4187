"""Compare linear and pairwise-quadratic trajectory rewards on the same users."""
import json
import sys

import numpy as np
import pandas as pd

from ecommerce_sim.history_trajectory_irl import QUADRATIC_NAMES, STATE_NAMES, evaluate, fit_maxent
from scripts.run_history_trajectory_irl import CAPSTONE, ROOT, windows_for_user
from src.analysis.masked_embedding_clusters import source_data


def run(source):
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, _ = source_data(source, path, sample_modulus=100)
    members = pd.read_parquet(ROOT / 'artifacts/three-ecommerce-personas' / f'{source}_memberships.parquet')
    if members.user_id.duplicated().any():
        raise ValueError('duplicate user assignments')
    selected = events[events.user.isin(members.user_id)].groupby('user', sort=False).head(64)
    split = members.set_index('user_id')['split'].to_dict()
    histories, paths = ({key: [] for key in ('train', 'test')} for _ in range(2))
    user_counts = {'train': 0, 'test': 0}
    for user, frame in selected.groupby('user', sort=False):
        h, p = windows_for_user(frame, source)
        if h:
            key = split[user]
            histories[key].extend(h)
            paths[key].extend(p)
            user_counts[key] += 1
    x = {k: np.asarray(histories[k], dtype=float).reshape(-1, 5) for k in histories}
    y = {k: np.asarray(paths[k], dtype=int).reshape(-1, 3) for k in paths}
    if not len(y['train']) or not len(y['test']):
        raise ValueError('empty train or test windows')
    models = {}
    for quadratic in (False, True):
        label = 'quadratic' if quadratic else 'linear'
        theta, diagnostic = fit_maxent(x['train'], y['train'], quadratic=quadratic)
        names = QUADRATIC_NAMES if quadratic else STATE_NAMES
        models[label] = {'terms': names, 'coefficients': dict(zip(('cart', 'purchase'),
                                                                  theta.reshape(2, -1).tolist())),
                         'train': evaluate(x['train'], y['train'], theta, quadratic=quadratic),
                         'test': evaluate(x['test'], y['test'], theta, quadratic=quadratic),
                         'optimizer': diagnostic}
        print(source, label, 'test NLL=', models[label]['test']['trajectory_nll_per_action'], flush=True)
    result = {'source': source, 'source_quality': quality, 'temperature': 1.0,
              'penalty': 1.0, 'counts_scale': 'log1p(count)/log(65)',
              'users_with_windows': user_counts, 'windows': {k: len(y[k]) for k in y},
              'models': models,
              'limitations': ['Same fixed hyperparameters for both models; no separate tuning or confidence intervals.',
                  'Retrospectively sampled users; full-history session counts influenced sampling.',
                  'Observed complete three-action segments only; no action availability, session termination or exposure.',
                  'Pooled per-dataset utility, not individual reward, causal uplift, or validated simulator.',
                  'Quadratic features refer to log-scaled state variables, not raw event counts.',
                  'View action is the zero-reward reference; all coefficients are relative and reward is not identified.']}
    out = ROOT / 'artifacts/quadratic-history-irl'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
