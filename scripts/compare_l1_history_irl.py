"""Compare quadratic L2 and quadratic elastic-net on the same users."""
import json
import sys

import numpy as np
import pandas as pd

from ecommerce_sim.models.history_trajectory_irl import LAG_NAMES, QUADRATIC_NAMES, evaluate, fit_maxent
from scripts.run_history_trajectory_irl import CAPSTONE, ROOT, windows_for_user


def run(source, l1_only=False, lags=False):
    from src.analysis.masked_embedding_clusters import source_data

    if lags and not l1_only:
        raise ValueError('lag comparison requires --l1-only')
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, _ = source_data(source, path, sample_modulus=100)
    members = pd.read_parquet(ROOT / 'artifacts/three-ecommerce-personas' / f'{source}_memberships.parquet')
    if members.user_id.duplicated().any():
        raise ValueError('duplicate user assignments')
    selected = events[events.user.isin(members.user_id)].groupby('user', sort=False).head(64)
    split = members.set_index('user_id')['split'].to_dict()
    histories, paths = ({key: [] for key in ('train', 'test')} for _ in range(2))
    users = {'train': 0, 'test': 0}
    for user, frame in selected.groupby('user', sort=False):
        h, p = windows_for_user(frame, source, include_lags=lags)
        if h:
            key = split[user]
            histories[key].extend(h)
            paths[key].extend(p)
            users[key] += 1
    x = {k: np.asarray(histories[k], dtype=float).reshape(-1, 7 if lags else 5) for k in histories}
    y = {k: np.asarray(paths[k], dtype=int).reshape(-1, 3) for k in paths}
    if not len(y['train']) or not len(y['test']):
        raise ValueError('empty train or test windows')
    previous_dir = 'artifacts/l1-only-history-irl' if lags else 'artifacts/quadratic-history-irl'
    previous = json.loads((ROOT / previous_dir / f'{source}.json').read_text())
    if previous['users_with_windows'] != users or previous['windows'] != {k: len(y[k]) for k in y}:
        raise ValueError('previous quadratic cohort differs from current cohort')
    models = ({'quadratic_l1_no_lags': previous['models']['quadratic_l1_only']} if lags else
              {'quadratic_l2': previous['models']['quadratic']})
    l2 = 0.0 if l1_only else 1.0
    label = 'quadratic_l1_lag2' if lags else 'quadratic_l1_only' if l1_only else 'quadratic_l1_l2'
    theta, diagnostic = fit_maxent(x['train'], y['train'], penalty=l2, quadratic=True, l1=1, lags=lags)
    models[label] = {'terms': QUADRATIC_NAMES + LAG_NAMES if lags else QUADRATIC_NAMES,
                     'coefficients': dict(zip(('cart', 'purchase'),
                                                                       theta.reshape(2, -1).tolist())),
                     'zero_coefficients': int(np.count_nonzero(np.abs(theta) < 1e-8)),
                     'train': evaluate(x['train'], y['train'], theta, quadratic=True, lags=lags),
                     'test': evaluate(x['test'], y['test'], theta, quadratic=True, lags=lags),
                     'optimizer': diagnostic}
    print(source, label, 'test NLL=', models[label]['test']['trajectory_nll_per_action'],
          'zeros=', models[label]['zero_coefficients'], flush=True)
    result = {'source': source, 'source_quality': quality, 'temperature': 1.0, 'l2_strength': l2,
              'l1_strength': 1.0, 'objective': 'sum trajectory NLL + L1*sum(abs(theta)) + L2/2*sum(theta^2)',
              'state_scale': 'z=log1p(count)/log(65); no additional logarithmic features',
              'lags': 'last two observed common actions within session; reset after unmodeled actions' if lags else None,
              'users_with_windows': users, 'windows': {k: len(y[k]) for k in y}, 'models': models,
              'limitations': ['Fixed exploratory penalty strengths; no held-out hyperparameter tuning.',
                  'Feature terms can be collinear, so individual coefficients can be unstable.',
                  'L1 sparsity depends on feature scales and regularization; no causal feature selection.',
                  'This test split was already used in prior exploratory model comparisons.',
                  'Observed complete three-action segments only, with no modeled action availability or session termination.',
                  'Pooled dataset-level behavioral parameters, not individual reward or causal effects.']}
    out = ROOT / ('artifacts/lag-history-irl' if lags else
                  'artifacts/l1-only-history-irl' if l1_only else 'artifacts/l1-history-irl')
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    l1_only = '--l1-only' in sys.argv[1:]
    lags = '--lags' in sys.argv[1:]
    for source in (arg for arg in sys.argv[1:] if arg not in ('--l1-only', '--lags')):
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source, l1_only=l1_only, lags=lags)
