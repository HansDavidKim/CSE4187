#!/usr/bin/env python3
"""Train user-wise one-step MaxEnt and evaluate observational choice simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from ecommerce_sim.choice import CalibratedMultinomialLogit
from ecommerce_sim.nts_adapter import load_sequences
from ecommerce_sim.user_choice import UserMaxEnt, make_states, state_spec, weekly_choices


def metrics(prob, actual):
    flat, y = prob.reshape(-1, 3), actual.ravel()
    onehot = np.eye(3)[y]
    return {'log_loss': float(log_loss(y, flat, labels=[0, 1, 2])),
            'brier': float(np.square(flat - onehot).sum(axis=1).mean()),
            'accuracy': float((flat.argmax(axis=1) == y).mean()),
            'macro_f1': float(f1_score(y, flat.argmax(axis=1), labels=[0, 1, 2], average='macro', zero_division=0)),
            'observed_action_counts': np.bincount(y, minlength=3).tolist(),
            'observed_action_rates': (np.bincount(y, minlength=3) / len(y)).tolist(),
            'mean_predicted_action_probability': flat.mean(axis=0).tolist()}


def calibrate(model, state, action, pooled=False):
    u = model.utilities(state, pooled=pooled)
    calibration = CalibratedMultinomialLogit().fit(u.reshape(-1, 3), action.ravel())
    return calibration


def probs(model, calibration, state, pooled=False):
    utility = model.utilities(state, pooled=pooled)
    return calibration.probabilities(utility.reshape(-1, 3)).reshape(*utility.shape)


def sequence_metrics(actions):
    previous = actions[..., :-1] > 0
    current = actions[..., 1:] > 0
    return {'action_switch_rate': float(np.mean(actions[..., 1:] != actions[..., :-1])),
            'purchase_after_purchase': float(current[previous].mean()) if previous.any() else None}


def rollout(model, calibration, train, full, features, spec, start, end, seed):
    """Replay recorded offer schedule; generate outcomes from action-matched training donors."""
    rng = np.random.default_rng(seed)
    n = len(train)
    simulated = np.copy(full[:, :end])
    choices = np.empty((n, end-start), dtype=int)
    train_actions = weekly_choices(train, features)
    global_donors = [np.argwhere(train_actions == a) for a in range(3)]
    local_donors = [[np.flatnonzero(train_actions[i] == a) for a in range(3)] for i in range(n)]
    exogenous = [features.index(f) for f in ('offers_started', 'unique_coupons_started', 'active_offers')]
    for t in range(start, end):
        state = make_states(simulated, features, spec, np.array([t]))
        p = probs(model, calibration, state)[:, 0, :]
        action = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(max=2)
        choices[:, t-start] = action
        for i, a in enumerate(action):
            indices = local_donors[i][a]
            if len(indices):
                donor = train[i, rng.choice(indices)]
            else:
                user, week = global_donors[a][rng.integers(len(global_donors[a]))]
                donor = train[user, week]
            simulated[i, t] = donor
        # Observed assignment proxies are replayed, not simulated interventions.
        simulated[:, t, exogenous] = full[:, t, exogenous]
    return choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_week_panel.csv.gz'))
    parser.add_argument('--dag', type=Path, default=Path('artifacts/amexpert-nts-train-2026/nts_dag.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-user-choice-pilot'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error('rollouts must be positive')
    sequences, features = load_sequences(args.sequences)
    if any(len(s) != 78 for s in sequences):
        raise ValueError('pilot split requires 78 weeks per customer')
    dag = json.loads(args.dag.read_text())
    spec = state_spec(dag)
    values = np.stack(sequences)
    panel = pd.read_csv(args.panel).sort_values(['customer_id', 'week'])
    grouped = panel.groupby('customer_id', sort=True)
    user_ids = grouped.size().index.to_numpy()
    if len(user_ids) != len(values) or not np.all(grouped.size().to_numpy() == values.shape[1]):
        raise ValueError('panel users and sequence offsets do not match')
    if not np.allclose(panel[features].to_numpy(), values.reshape(-1, len(features))):
        raise ValueError('panel order/values do not match sequence archive')
    actions = weekly_choices(values, features)
    train_t, val_t, test_t = np.arange(2, 52), np.arange(52, 65), np.arange(65, 78)
    train_x, val_x, test_x = [make_states(values, features, spec, t) for t in (train_t, val_t, test_t)]
    train_y, val_y, test_y = [actions[:, t] for t in (train_t, val_t, test_t)]
    candidates = []
    for penalty in (2.0, 10.0, 50.0):
        model = UserMaxEnt(penalty).fit(train_x, train_y)
        calibration = calibrate(model, val_x, val_y)
        score = metrics(probs(model, calibration, val_x), val_y)['log_loss']
        candidates.append((score, penalty, model, calibration))
        print(f'penalty={penalty:g} validation_log_loss={score:.5f}', flush=True)
    candidates.sort(key=lambda item: item[0])
    val_loss, penalty, model, calibration = candidates[0]
    pooled_calibration = calibrate(model, val_x, val_y, pooled=True)
    test_prob = probs(model, calibration, test_x)
    pooled_prob = probs(model, pooled_calibration, test_x, pooled=True)
    train_counts = np.stack([(train_y == a).sum(axis=1) for a in range(3)], axis=1)
    global_rates = np.bincount(train_y.ravel(), minlength=3) / train_y.size
    frequency = (train_counts + 10 * global_rates) / (train_y.shape[1] + 10)
    baseline = np.repeat(frequency[:, None, :], len(test_t), axis=1)
    rollout_choices = np.stack([rollout(model, calibration, values[:, :52], values, features, spec, 65, 78, args.seed + i) for i in range(args.rollouts)])
    observed_rates = np.stack([(test_y == a).mean(axis=1) for a in range(3)], axis=1)
    generated_rates = np.stack([(rollout_choices == a).mean(axis=(0, 2)) for a in range(3)], axis=1)
    summary = {'method': 'user-specific regularized one-step contextual MaxEnt reward + calibrated multinomial logit',
        'actions': ['no_purchase', 'purchase_without_recorded_coupon_discount', 'purchase_with_recorded_coupon_discount'],
        'data': {'customers': len(values), 'train_weeks': [0, 51], 'validation_weeks': [52, 64], 'test_weeks': [65, 77],
                 'note': 'first two train weeks supply lag history; graph training cutoff must be checked separately'},
        'source_dag': str(args.dag), 'state_features': [{'feature': f, 'lag': lag} for f, lag in spec],
        'non_graph_control_features': ['transactions lag1', 'active_offers lag1'],
        'regularization_validation_logloss': {str(p): score for score, p, _, _ in candidates},
        'chosen_regularization': penalty, 'optimization_iterations': int(model.optimization_iterations_),
        'choice_temperature_diagnostics': calibration.diagnostics(model.utilities(val_x).reshape(-1, 3)),
        'test_one_step_user_model': metrics(test_prob, test_y),
        'test_one_step_pooled_model': metrics(pooled_prob, test_y),
        'test_one_step_user_frequency_baseline': metrics(baseline, test_y),
        'test_simulation': {'rollouts': args.rollouts,
            'observed_action_rates': (np.bincount(test_y.ravel(), minlength=3) / test_y.size).tolist(),
            'generated_action_rates': (np.bincount(rollout_choices.ravel(), minlength=3) / rollout_choices.size).tolist(),
            'user_action_rate_mae': np.abs(observed_rates-generated_rates).mean(axis=0).tolist(),
            'observed_sequence': sequence_metrics(test_y),
            'generated_sequence': sequence_metrics(rollout_choices)},
        'limitations': ['one-step contextual MaxEnt, not long-horizon IRL',
            'weekly observed outcome categories are behavioral proxies, not observed item alternatives or exposure choices',
            'coupon discount is not equivalent to redemption', 'historical offer schedule replayed; not intervention evaluation',
            'training-outcome donor resampling is a heuristic, not an identified structural transition model',
            'same customer training and test; no cold-start validation', 'NTS graph is an observational hypothesis',
            'local derivatives should not be redistributed before data rights review']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(args.output / 'local_user_rewards.npz', user_ids=user_ids, weights=model.weights_, mean=model.mean_, scale=model.scale_,
                        state_features=np.array([f'{f}_lag{lag}' for f, lag in spec]),
                        penalty=penalty, temperature=calibration.temperature_, utility_scale=calibration.utility_scale_)
    np.savez_compressed(args.output / 'local_simulated_choices.npz', user_ids=user_ids, choices=rollout_choices,
                        labels=np.array(summary['actions']), seed=args.seed)
    print(json.dumps({'chosen_regularization': penalty, 'test_user': summary['test_one_step_user_model'],
                      'test_pooled': summary['test_one_step_pooled_model'],
                      'simulation': summary['test_simulation']}, indent=2))


if __name__ == '__main__':
    main()
