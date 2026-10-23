#!/usr/bin/env python3
"""Compare linear and neural utilities under the same trajectory IRL MDP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ecommerce_sim.nts_adapter import load_sequences
from ecommerce_sim.nonlinear_trajectory_irl import fit_reward, fit_transitions, make_states, policy_numpy
from ecommerce_sim.user_choice import weekly_choices


def score(policy, groups, states, actions):
    p = policy[groups[:, None], np.arange(actions.shape[1])[None, :], states]
    selected = np.take_along_axis(p, actions[..., None], axis=-1)[..., 0]
    return {'log_loss': float(-np.log(np.maximum(selected, 1e-12)).mean()),
            'brier': float(((p - np.eye(3)[actions]) ** 2).sum(axis=-1).mean()),
            'accuracy': float((p.argmax(-1) == actions).mean()),
            'mean_entropy': float((-p * np.log(np.maximum(p, 1e-12))).sum(axis=-1).mean())}


def rollout(policy, transition, groups, initial, n, seed, greedy=False):
    rng = np.random.default_rng(seed)
    paths = np.empty((n, len(groups), policy.shape[1]), dtype=int)
    for k in range(n):
        state = initial.copy()
        for t in range(policy.shape[1]):
            p = policy[groups, t, state]
            a = p.argmax(-1) if greedy else np.minimum((rng.random(len(groups))[:, None] > np.cumsum(p, axis=1)).sum(axis=1), 2)
            paths[k, :, t] = a
            q = transition[state, a]
            state = np.minimum((rng.random(len(groups))[:, None] > np.cumsum(q, axis=1)).sum(axis=1), 11)
    return paths


def fidelity(paths, actual):
    rates = np.stack([(paths == i).mean(axis=(0, 2)) for i in range(3)], axis=1)
    observed = np.stack([(actual == i).mean(axis=1) for i in range(3)], axis=1)
    before, after = paths[..., :-1] > 0, paths[..., 1:] > 0
    return {'action_rates': (np.bincount(paths.ravel(), minlength=3) / paths.size).tolist(),
            'user_rate_mae': np.abs(rates-observed).mean(axis=0).tolist(),
            'switch_rate': float((paths[..., 1:] != paths[..., :-1]).mean()),
            'purchase_after_purchase': float(after[before].mean()) if before.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_week_panel.csv.gz'))
    parser.add_argument('--groups', type=Path, default=Path('artifacts/amexpert-group-trajectory-irl/local_groups_rewards.npz'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-nonlinear-trajectory-irl'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error('rollouts must be positive')
    torch.set_num_threads(1)
    sequences, features = load_sequences(args.sequences)
    values = np.stack(sequences)
    with np.load(args.groups, allow_pickle=False) as saved:
        groups, ids = saved['groups'], saved['user_ids']
    if values.shape[1] != 78 or len(groups) != len(values):
        raise ValueError('group membership must match the 78-week panel')
    panel = pd.read_csv(args.panel).sort_values(['customer_id', 'week'])
    grouped = panel.groupby('customer_id', sort=True)
    if not np.array_equal(grouped.size().index.to_numpy(), ids) or not np.all(grouped.size().to_numpy() == 78):
        raise ValueError('group IDs or sequence lengths do not match the panel')
    if not np.allclose(panel[features].to_numpy(), values.reshape(-1, len(features))):
        raise ValueError('panel data do not match the sequence archive')
    actions = weekly_choices(values, features)
    quantity = values[:, :, features.index('quantity')]
    offers = values[:, :, features.index('active_offers')]
    positives = quantity[:, :52][quantity[:, :52] > 0]
    threshold = float(np.median(positives))
    states = make_states(actions, quantity, offers, threshold)
    transition = fit_transitions(states, actions)
    starts = (1, 14, 27, 39)
    train_states = np.stack([states[:, t:t+13] for t in starts], axis=1)
    train_actions = np.stack([actions[:, t:t+13] for t in starts], axis=1)
    test_y = actions[:, 65:78]
    observed = fidelity(test_y[None], test_y)
    results, arrays = {}, {}
    for nonlinear in (False, True):
        name = 'neural' if nonlinear else 'linear'
        reward, fit = [], []
        for g in range(int(groups.max()) + 1):
            rows = groups == g
            if not rows.any():
                raise ValueError('empty user group')
            r, info = fit_reward(train_states[rows].reshape(-1, 13), train_actions[rows].reshape(-1, 13),
                                 transition, nonlinear, args.seed + g)
            reward.append(r)
            fit.append(info)
        policies = np.stack([policy_numpy(r, transition, 13) for r in reward])
        simulated = rollout(policies, transition, groups, states[:, 65], args.rollouts, args.seed)
        greedy = rollout(policies, transition, groups, states[:, 65], args.rollouts, args.seed, greedy=True)
        results[name] = {'fit': fit,
            'training_teacher_forced': score(policies, np.repeat(groups, len(starts)),
                                             train_states.reshape(-1, 13), train_actions.reshape(-1, 13)),
            'validation_teacher_forced': score(policies, groups, states[:, 52:65], actions[:, 52:65]),
            'test_teacher_forced': score(policies, groups, states[:, 65:78], test_y),
            'stochastic_rollout': fidelity(simulated, test_y), 'greedy_rollout': fidelity(greedy, test_y)}
        arrays[name+'_reward'] = np.asarray(reward)
        arrays[name+'_policy'] = policies
        arrays[name+'_simulated'] = simulated
        print(name, json.dumps(results[name]['test_teacher_forced']), flush=True)
    summary = {'method': 'finite-horizon maximum causal entropy IRL: linear vs tanh MLP reward',
        'group_source': str(args.groups), 'seed': args.seed, 'horizon': 13,
        'state': 'previous weekly action (3) x previous quantity above training median positive quantity (2) x previous active offer >0 (2)',
        'quantity_threshold': threshold, 'transition': 'pooled observational P(s_next|s,action), additive smoothing 0.5; estimated on weeks 1..50',
        'train_episodes': [[1,13],[14,26],[27,39],[39,51]], 'validation_weeks': [52,64], 'test_weeks': [65,77],
        'observed_test': observed, 'results': results,
        'limitations': ['Same fixed GMM groups and transition model for both utility families.',
          'Reward is not uniquely identifiable; one global reward reference and regularization fix a convention.',
          'Neural network is unconstrained, so no economic monotonicity or willingness-to-pay is established.',
          'Observed discount category is a behavioral proxy, not an observed choice set or guaranteed available action.',
          'Transitions are observational and pooled; they are not identified under interventions.',
          'Same regularization 1 for both models; no hyperparameter or seed robustness study.',
          'Group memberships and sequence ordering are checked against the local customer panel.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(args.output / 'local_models_and_paths.npz', user_ids=ids, groups=groups, transition=transition, **arrays)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
