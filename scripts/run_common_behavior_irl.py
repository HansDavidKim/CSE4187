"""Fit per-cluster finite-horizon MaxEnt IRL on common observed event windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecommerce_sim.models.common_behavior_irl import COEFFICIENT_NAMES, trajectory_windows
from ecommerce_sim.models.trajectory_irl import fit_trajectory_reward, policy_numpy
from ecommerce_sim.paths import CAPSTONE_ROOT


CAPSTONE = CAPSTONE_ROOT
ROOT = Path('artifacts/three-ecommerce-personas')
OUT = Path('artifacts/common-behavior-irl')
HORIZON = 3
BOOTSTRAPS = 5


def join_windows(windows, indices):
    selected = [windows[i] for i in indices if len(windows[i][0])]
    if not selected:
        return np.empty((0, HORIZON), dtype=int), np.empty((0, HORIZON), dtype=int)
    return np.concatenate([x[0] for x in selected]), np.concatenate([x[1] for x in selected])


def log_loss(reward, states, actions):
    if len(states) == 0:
        return None
    policy = policy_numpy(reward, HORIZON)
    p = policy[np.arange(HORIZON)[None, :], states, actions]
    return float(-np.log(np.maximum(p, 1e-12)).mean())


def run(source):
    from src.analysis.masked_embedding_clusters import rows_from_events, source_data

    torch.set_num_threads(1)
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    members = pd.read_parquet(ROOT / f'{source}_memberships.parquet')
    if members.user_id.tolist() != [row[0] for row in rows]:
        raise ValueError('Sample differs from saved cluster assignments')
    stability = json.loads((ROOT / f'{source}_stability.json').read_text())
    groups = stability['results'][4]['groups']  # k=6, same partition within source
    kept = [g['cluster'] for g in groups if g['heldout_users'] >= 50 and g['min_jaccard'] >= .8]
    windows = [trajectory_windows(row[1], actions, HORIZON) for row in rows]
    train_mask = (members['split'].eq('train') & members.full_history_sessions.ge(2)).to_numpy()
    test_mask = (members['split'].eq('test') & members.full_history_sessions.ge(2)).to_numpy()
    rng = np.random.default_rng(2026)
    output = []
    for group in kept:
        is_group = members.k6.to_numpy() == group
        train = np.flatnonzero(train_mask & is_group)
        test = np.flatnonzero(test_mask & is_group)
        informative_train = np.array([i for i in train if len(windows[i][0])])
        informative_test = np.array([i for i in test if len(windows[i][0])])
        states, choices = join_windows(windows, informative_train)
        test_states, test_choices = join_windows(windows, informative_test)
        if not len(states):
            output.append({'cluster': int(group), 'status': 'no_complete_windows',
                           'train_users': len(train), 'test_users': len(test)})
            continue
        reward, diagnostic = fit_trajectory_reward(states, choices, penalty=1.0)
        samples = []
        for _ in range(BOOTSTRAPS):
            resampled = rng.choice(informative_train, len(informative_train), replace=True)
            s, a = join_windows(windows, resampled)
            value, _ = fit_trajectory_reward(s, a, penalty=1.0)
            samples.append(value.ravel()[1:].tolist())
        matrix = np.array(samples)
        output.append({'cluster': int(group), 'status': 'fit',
                       'train_users': len(train), 'train_users_with_windows': len(informative_train),
                       'train_windows': len(states), 'test_users': len(test),
                       'test_users_with_windows': len(informative_test), 'test_windows': len(test_states),
                       'reward_matrix': reward.tolist(), 'coefficients': reward.ravel()[1:].tolist(),
                       'bootstrap_coefficients': samples,
                       'bootstrap_std': matrix.std(axis=0, ddof=1).tolist(),
                       'train_log_loss': log_loss(reward, states, choices),
                       'heldout_descriptive_log_loss': log_loss(reward, test_states, test_choices),
                       'optimizer': diagnostic})
        print(source, 'cluster', group, 'train users/windows', len(informative_train), len(states),
              'test windows', len(test_states), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {'source': source, 'quality': quality, 'horizon': HORIZON, 'k': 6,
               'feature_names': list(COEFFICIENT_NAMES), 'anchor': 'view_to_view = 0',
               'penalty': 1.0, 'temperature': 1.0, 'bootstraps': BOOTSTRAPS,
               'selection': 'heldout users >=50 and min stability Jaccard >=0.8; post-hoc screen',
               'limitations': ['Condition on observed fixed-length event windows; not a session generator.',
                   'Remove and session end split windows, not modeled as actions.',
                   'Test cluster uses same full input prefix as test windows, so not prospective prediction.',
                   'Clusters and screening reused test data; descriptive test only.',
                   'Rewards are regularized, scale/shape non-identified; not individual preferences.',
                   'Bootstrap is uncertainty of cluster estimate, not between-user coefficient distribution.',
                   'No causal treatment or uplift is estimated.'], 'groups': output}
    (OUT / f'{source}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'electronics', 'retailrocket', 'cosmetics'}:
        raise SystemExit('Specify electronics, retailrocket or cosmetics')
    run(sys.argv[1])
