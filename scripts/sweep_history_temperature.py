"""Calibrate trajectory logits on held-out users, then evaluate once on test users."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ecommerce_sim.history_trajectory_irl import ACTIONS, PATHS, STATE_NAMES, feature_counts, fit_maxent
from scripts.run_history_trajectory_irl import CAPSTONE, ROOT, windows_for_user
from src.analysis.masked_embedding_clusters import source_data

TEMPERATURES = sorted(set([0.5, 0.6, 0.7, 0.8, 1.2, 1.4, 1.6, 2.0] +
                          [round(x / 1000, 3) for x in range(825, 1200, 25)]))


def measures(scores, observed, temperature):
    if temperature <= 0 or scores.shape != (len(observed), len(PATHS)) or not len(observed):
        raise ValueError('positive temperature and nonempty aligned trajectory scores required')
    scaled = scores / temperature
    logp = scaled - logsumexp(scaled, axis=1, keepdims=True)
    p = np.exp(logp)
    indices = observed[:, 0] * 9 + observed[:, 1] * 3 + observed[:, 2]
    observed_shares, predicted_shares = [], []
    for position in range(3):
        observed_shares.append((np.bincount(observed[:, position], minlength=3) / len(observed)).tolist())
        predicted_shares.append([float(p[:, PATHS[:, position] == a].sum() / len(observed)) for a in range(3)])
    first_predicted = np.array(predicted_shares[0])
    first_observed = np.array(observed_shares[0])
    return {'trajectory_nll_per_action': float(-logp[np.arange(len(indices)), indices].mean() / 3),
            'trajectory_entropy': float(-(p * logp).sum(axis=1).mean()),
            'observed_action_shares_by_position': observed_shares,
            'predicted_action_shares_by_position': predicted_shares,
            'first_action_total_variation': float(np.abs(first_observed - first_predicted).sum() / 2)}


def run(source):
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, _ = source_data(source, path, sample_modulus=100)
    members = pd.read_parquet(ROOT / 'artifacts/three-ecommerce-personas' / f'{source}_memberships.parquet')
    if members.user_id.duplicated().any():
        raise ValueError('duplicate user assignments')
    original_train = members.loc[members.split.eq('train'), 'user_id'].to_numpy()
    rng = np.random.default_rng(2026)
    validation = set(rng.choice(original_train, size=len(original_train) // 5, replace=False))
    split = {row.user_id: ('validation' if row.user_id in validation else row.split)
             for row in members.itertuples()}
    selected = events[events.user.isin(members.user_id)].groupby('user', sort=False).head(64)
    names = ('train', 'validation', 'test')
    histories, paths = ({name: [] for name in names} for _ in range(2))
    users = dict.fromkeys(names, 0)
    for user, frame in selected.groupby('user', sort=False):
        h, p = windows_for_user(frame, source)
        if h:
            group = split[user]
            histories[group].extend(h)
            paths[group].extend(p)
            users[group] += 1
    x = {k: np.asarray(histories[k], dtype=float).reshape(-1, 5) for k in names}
    y = {k: np.asarray(paths[k], dtype=int).reshape(-1, 3) for k in names}
    if any(not len(y[k]) for k in names):
        raise ValueError('insufficient windows for train, validation, or test')
    theta, diagnostic = fit_maxent(x['train'], y['train'])
    validation_scores = feature_counts(x['validation']) @ theta
    sweep = [{'temperature': t, **measures(validation_scores, y['validation'], t)} for t in TEMPERATURES]
    best = min(sweep, key=lambda row: row['trajectory_nll_per_action'])['temperature']
    best_first = min(sweep, key=lambda row: row['first_action_total_variation'])['temperature']
    test_scores = feature_counts(x['test']) @ theta
    result = {'source': source, 'source_quality': quality, 'actions': ACTIONS, 'states': STATE_NAMES,
              'model': 'MaxEnt three-step trajectory logits = candidate cumulative reward / temperature',
              'coefficients_cart_purchase': theta.reshape(2, -1).tolist(),
              'split_seed': 2026, 'validation_fraction_of_original_train_users': 0.2,
              'users_with_windows': users, 'windows': {k: len(y[k]) for k in names},
              'optimizer': diagnostic, 'validation_sweep': sweep,
              'selection_metric': 'validation trajectory NLL per action', 'selected_temperature': best,
              'selected_temperature_for_first_action_tv': best_first,
              'test_unscaled': measures(test_scores, y['test'], 1.0),
              'test_selected': measures(test_scores, y['test'], best),
              'test_first_action_tv_selected': measures(test_scores, y['test'], best_first),
              'limitations': ['T scales logits of whole three-action paths, not a single immediate-action logit.',
                  'Temperature selection optimizes likelihood, not first-action or persona matching.',
                  'Test outcomes did not select temperature; validation and test users are disjoint.',
                  'Users were retrospectively sampled using full-history session counts.',
                  'Three-action windows only; action availability, termination and exposures are not modeled.',
                  'No persona-specific or user-specific reward; no claims of causal or intervention effects.']}
    out = ROOT / 'artifacts/history-temperature-sweep'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, 'T=', best, 'users=', users, 'windows=', result['windows'],
          'test nll=', result['test_unscaled']['trajectory_nll_per_action'], '->',
          result['test_selected']['trajectory_nll_per_action'], flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
