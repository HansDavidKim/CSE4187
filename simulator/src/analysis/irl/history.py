"""History-based trajectory IRL experiments and recent-feature audits."""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from models.history_trajectory_irl import (
    ACTIONS, LAG_NAMES, PATHS, QUADRATIC_NAMES, STATE_NAMES, evaluate,
    feature_counts, fit_maxent,
)
from preprocessing.recent_history_features import (
    ACTIONS as RECENT_ACTIONS, decision_rows, feature_names,
)
from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT

CAPSTONE = CAPSTONE_ROOT
ROOT = PROJECT_ROOT


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


def run_fit(source):
    from analysis.persona.sequence import source_data

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


def run_l1(source, l1_only=False, lags=False):
    from analysis.persona.sequence import source_data

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


def run_quadratic(source):
    from analysis.persona.sequence import source_data

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


def run_temperature(source):
    from analysis.persona.sequence import source_data

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


def run_feature_audit(source):
    from analysis.persona.sequence import source_data

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
    result = {'source': source, 'source_quality': quality, 'actions': RECENT_ACTIONS,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('fit', 'quadratic', 'l1', 'temperature', 'features'))
    parser.add_argument('sources', nargs='+', choices=('electronics', 'cosmetics', 'retailrocket'))
    parser.add_argument('--l1-only', action='store_true')
    parser.add_argument('--lags', action='store_true')
    args = parser.parse_args()
    if args.task != 'l1' and (args.l1_only or args.lags):
        parser.error('--l1-only and --lags apply only to the l1 task')
    for source in args.sources:
        if args.task == 'l1':
            run_l1(source, l1_only=args.l1_only, lags=args.lags)
        else:
            {'fit': run_fit, 'quadratic': run_quadratic,
             'temperature': run_temperature, 'features': run_feature_audit}[args.task](source)


if __name__ == '__main__':
    main()
