"""Train common-history MaxEnt IRL; outputs contain only aggregate statistics."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_sim.history_trajectory_irl import ACTIONS, STATE_NAMES, evaluate, fit_maxent
from src.analysis.masked_embedding_clusters import source_data

CAPSTONE = Path('/Users/kimdaewon/Desktop/캡스톤디자인')
ROOT = Path('/Users/kimdaewon/Desktop/eCommerce-Sim')


def windows_for_user(frame, source, include_lags=False):
    """Disjoint three-event windows; unsupported events and sessions break paths."""
    history, observed = [], []
    prior_sessions = prior_purchases = 0
    action_map = {0: 0, 1: 1, 3: 2} if source == 'cosmetics' else {0: 0, 1: 1, 2: 2}
    for _, session in frame.groupby('session', sort=False):
        counts = np.zeros(3, dtype=int)
        segment = []
        previous_actions = []
        start = None
        for raw in session.action:
            if raw not in action_map:
                segment, start = [], None
                previous_actions = []
                continue
            action = action_map[raw]
            if not segment:
                start = [*counts.tolist(), prior_sessions, prior_purchases]
                if include_lags:
                    start.extend([previous_actions[-1] if previous_actions else -1,
                                  previous_actions[-2] if len(previous_actions) > 1 else -1])
            segment.append(action)
            counts[action] += 1
            previous_actions.append(action)
            previous_actions = previous_actions[-2:]
            if len(segment) == 3:
                history.append(start)
                observed.append(segment)
                segment, start = [], None
        prior_sessions += 1
        prior_purchases += int(counts[2])
    return history, observed


def run(source):
    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics'
                       else f'data/raw/kaggle/{source}/events.csv')
    events, quality, _ = source_data(source, path, sample_modulus=100)
    members = pd.read_parquet(ROOT / 'artifacts/three-ecommerce-personas' / f'{source}_memberships.parquet')
    if members.user_id.duplicated().any():
        raise ValueError('duplicate user assignments')
    selected = events[events.user.isin(members.user_id)].copy()
    selected = selected.groupby('user', sort=False).head(64)
    split = members.set_index('user_id')['split'].to_dict()
    histories = {'train': [], 'test': []}
    paths = {'train': [], 'test': []}
    users = {'train': 0, 'test': 0}
    for user, frame in selected.groupby('user', sort=False):
        h, p = windows_for_user(frame, source)
        if h:
            group = split[user]
            histories[group].extend(h)
            paths[group].extend(p)
            users[group] += 1
    x_train, x_test = (np.asarray(histories[k], dtype=float).reshape(-1, 5) for k in ('train', 'test'))
    y_train, y_test = (np.asarray(paths[k], dtype=int).reshape(-1, 3) for k in ('train', 'test'))
    if not len(y_train) or not len(y_test):
        raise ValueError('insufficient windows')
    theta, diagnostic = fit_maxent(x_train, y_train)
    counts = np.bincount(y_train.ravel(), minlength=3) + 1
    baseline = counts / counts.sum()
    result = {
        'source': source, 'source_quality': quality, 'actions': ACTIONS,
        'states': STATE_NAMES, 'scale': 'log1p(count)/log(65), intercept unchanged',
        'coefficient_reference': 'view',
        'coefficients': dict(zip(ACTIONS[1:], theta.reshape(2, -1).tolist())),
        'train_users_with_windows': users['train'], 'test_users_with_windows': users['test'],
        'selected_users': len(members), 'train_windows': len(y_train), 'test_windows': len(y_test),
        'train': evaluate(x_train, y_train, theta), 'test': evaluate(x_test, y_test, theta),
        'constant_baseline_test_nll_per_action': float(-np.log(baseline[y_test]).mean()),
        'optimizer': diagnostic,
        'limitations': [
            'User sample is stratified on full-history session counts in prior persona work; retrospective selection.',
            'Only observed complete three-action segments; not a model of session end, return, product, or exposure.',
            'Unobserved/remove events break windows, and unmodeled action availability can bias coefficients.',
            'Maximum 64 events per user; grouped train/test users but not temporal extrapolation.',
            'Timestamp ties preserve input order; exact within-tie order is not observed.',
            'Pooled dataset-specific rewards, not individual rewards, causal effects, or uplift.',
            'History-count dynamics are assumed deterministic; fixed horizon and reward gauge do not identify true utility.',
            'Evaluation is conditional on complete windows; it is not full-session simulation.'
        ]}
    out = ROOT / 'artifacts/history-trajectory-irl'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, users, len(y_train), len(y_test), result['test'], flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'retailrocket', 'cosmetics'):
            raise ValueError(source)
        run(source)
