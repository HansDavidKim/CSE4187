#!/usr/bin/env python3
"""Fit purchase trajectories conditional on a separately recorded offer proxy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ecommerce_sim.models.conditional_trajectory_irl import fit_reward, offer_transition, probabilities, sample_coefficients
from ecommerce_sim.preprocessing.nts_adapter import load_sequences
if __package__:
    from scripts.run_group_trajectory_irl import train_encoder
else:
    from run_group_trajectory_irl import train_encoder


def metrics(p, y):
    return {'log_loss': float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))),
            'brier': float(np.mean((p-y)**2)), 'purchase_rate': float(y.mean()),
            'predicted_purchase_rate': float(p.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_week_panel.csv.gz'))
    parser.add_argument('--dag', type=Path, default=Path('artifacts/amexpert-nts-train-2026/nts_dag.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-conditional-trajectory-irl'))
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.groups < 1 or args.epochs < 1:
        parser.error('groups and epochs must be positive')
    torch.set_num_threads(1)
    sequences, features = load_sequences(args.sequences)
    values = np.stack(sequences)
    if values.shape[1] != 78 or not np.isfinite(values).all():
        raise ValueError('expected finite 78-week panel')
    panel = pd.read_csv(args.panel).sort_values(['customer_id', 'week'])
    grouped = panel.groupby('customer_id', sort=True)
    user_ids = grouped.size().index.to_numpy()
    if len(user_ids) != len(values) or not np.all(grouped.size().to_numpy() == 78) or not np.allclose(panel[features].to_numpy(), values.reshape(-1, len(features))):
        raise ValueError('panel does not match the sequence archive')
    graph = json.loads(args.dag.read_text())
    if graph.get('converged') is not True or graph.get('instantaneous_is_dag') is not True:
        raise ValueError('expected converged acyclic NTS graph')
    # Only lagged transaction parents are potentially pre-decision. Never use same-week outcomes.
    graph_parents = sorted({e['source'] for e in graph['edges'] if e['target'] == 'transactions'
                            and int(e['lag']) > 0 and e['source'] in features})
    encoder_features = sorted(set(graph_parents) | {'transactions', 'active_offers'})
    raw = np.log1p(np.maximum(values[:, :52, [features.index(f) for f in encoder_features]], 0))
    mean, scale = raw.mean(axis=(0, 1)), raw.std(axis=(0, 1))
    scale[scale == 0] = 1
    _, embeddings = train_encoder((raw-mean)/scale, args.seed, args.epochs)
    embed_scale = StandardScaler().fit(embeddings)
    gmm = GaussianMixture(n_components=args.groups, reg_covar=1e-4, random_state=args.seed, n_init=3)
    groups = gmm.fit_predict(embed_scale.transform(embeddings))
    if len(np.unique(groups)) != args.groups:
        raise ValueError('empty group')
    purchase = (values[:, :, features.index('transactions')] > 0).astype(int)
    offers = (values[:, :, features.index('active_offers')] > 0).astype(int)
    offer_p = offer_transition(offers[:, :49])
    starts, horizon = (1, 13, 25, 37), 12
    state = np.stack([purchase[:, t-1:t+horizon-1] for t in starts], axis=1).reshape(-1, horizon)
    action = np.stack([purchase[:, t:t+horizon] for t in starts], axis=1).reshape(-1, horizon)
    offer = np.stack([offers[:, t:t+horizon] for t in starts], axis=1).reshape(-1, horizon)
    n = len(values)
    row_users = np.repeat(np.arange(n), len(starts))
    row_groups = groups[row_users]
    group_weights, group_fit = fit_reward(state, offer, action, offer_p, penalty=10., row_group=row_groups)
    user_weights, user_fit = fit_reward(state, offer, action, offer_p, center=group_weights[groups],
                                        penalty=10., row_group=row_users)
    test_h = 13
    t = np.arange(test_h)[None, :]
    rows = np.arange(n)[:, None]
    validation_s, validation_z, validation_y = purchase[:, 51:64], offers[:, 52:65], purchase[:, 52:65]
    validation_uncalibrated = probabilities(user_weights, offer_p, test_h)[rows, t, validation_s, validation_z, 1]

    def validation_loss(offset):
        shifted = user_weights.copy()
        shifted[:, [0, 2]] += offset
        pv = probabilities(shifted, offer_p, test_h)[rows, t, validation_s, validation_z, 1]
        return metrics(pv, validation_y)['log_loss']

    calibration = minimize(validation_loss, np.zeros(2), method='L-BFGS-B', bounds=[(-3., 3.), (-3., 3.)])
    if not calibration.success:
        raise RuntimeError('validation calibration failed')
    user_weights[:, [0, 2]] += calibration.x
    group_weights[:, [0, 2]] += calibration.x
    z, y, s = offers[:, 65:78], purchase[:, 65:78], purchase[:, 64:77]
    p_user = probabilities(user_weights, offer_p, test_h)[rows, t, s, z, 1]
    p_group = probabilities(group_weights, offer_p, test_h)[groups[:, None], t, s, z, 1]
    rng = np.random.default_rng(args.seed)
    policies = probabilities(user_weights, offer_p, test_h)
    sims = np.empty((100, n, test_h), dtype=np.int8)
    for k in range(len(sims)):
        current = purchase[:, 64].copy()
        for j in range(test_h):
            prob = policies[np.arange(n), j, current, z[:, j], 1]
            current = (rng.random(n) < prob).astype(int)
            sims[k, :, j] = current
    rates = {str(v): metrics(p_user[z == v], y[z == v]) for v in (0, 1) if np.any(z == v)}
    covariances = np.stack([np.atleast_2d(np.cov(user_weights[groups == i], rowvar=False))
                            if (groups == i).sum() > 1 else np.eye(4)*1e-6 for i in range(args.groups)])
    mixture_weights = np.bincount(groups, minlength=args.groups)/n
    mixture_means = np.stack([user_weights[groups == i].mean(axis=0) for i in range(args.groups)])
    synthetic_weights, synthetic_groups = sample_coefficients(mixture_weights, mixture_means, covariances, n, args.seed)
    synthetic_policies = probabilities(synthetic_weights, offer_p, test_h)
    synthetic_choices = np.empty((n, test_h), dtype=np.int8)
    synthetic_state = purchase[:, 64].copy()
    for j in range(test_h):
        p = synthetic_policies[np.arange(n), j, synthetic_state, z[:, j], 1]
        synthetic_state = (rng.random(n) < p).astype(int)
        synthetic_choices[:, j] = synthetic_state
    p_val = policies[rows, t, validation_s, validation_z, 1]
    baseline_p = np.full_like(p_user, purchase[:, 1:49].mean(), dtype=float)
    summary = {'model': 'finite-horizon MaxEnt purchase model conditional on recorded active offer proxy',
       'behavior': ['no_purchase', 'purchase'], 'intervention_proxy': 'recorded active offer this week; 0 means no record, not proven untreated',
       'features': {'irl_state': ['previous_week_purchase', 'current_recorded_offer'],
                    'reward_purchase_features': ['intercept', 'previous_week_purchase', 'current_recorded_offer', 'interaction'],
                    'lagged_nts_transaction_parents': graph_parents, 'encoder_features': encoder_features},
       'graph_limitation': 'NTS graph has no lagged parents for transactions' if not graph_parents else 'NTS observational edges are hypotheses, not identified causes',
       'train_weeks_encoder': [0, 51], 'irl_train_episodes': [[t, t+horizon-1] for t in starts],
       'test_weeks': [65, 77], 'group_sizes': np.bincount(groups, minlength=args.groups).tolist(),
       'offer_transition_training': offer_p.tolist(), 'group_fit': group_fit, 'user_fit': user_fit,
       'training_purchase_rate': float(purchase[:, 1:49].mean()),
       'training_recorded_offer_rate': float(offers[:, 1:49].mean()),
       'validation_recorded_offer_rate': float(validation_z.mean()), 'test_recorded_offer_rate': float(z.mean()),
       'calibration': {'method': 'shared purchase intercept and recorded-offer coefficient selected on validation log loss only',
                       'offsets': calibration.x.tolist(), 'validation_before': metrics(validation_uncalibrated, validation_y)},
       'validation_teacher_forced_user': metrics(p_val, validation_y),
       'teacher_forced_user': metrics(p_user, y), 'teacher_forced_group': metrics(p_group, y),
       'teacher_forced_constant_training_baseline': metrics(baseline_p, y),
       'user_by_recorded_offer': rates, 'test_observed_purchase_rate': float(y.mean()),
       'rollout_purchase_rate_mean': float(sims.mean()), 'rollout_purchase_rate_std': float(sims.mean(axis=(1, 2)).std()),
       'sampled_population_purchase_rate': float(synthetic_choices.mean()),
       'rollout_assumption': 'Replay recorded test-week offer proxy; user purchase and previous purchase simulated',
       'limitations': ['No recorded offer is not verified non-exposure; zeros mean no record only.',
         'Current-week record is a campaign-validity proxy, not verified pre-purchase exposure time.',
         'Offer assignment is observational and potentially confounded; no causal or uplift claim.',
         'NTS graph has no lagged purchase parents; previous purchase and current recorded offer are explicit assumptions.',
         'Binary current offer ignores offer type and within-week ordering.',
         'Future offer record transition is estimated observationally; independent of prior purchases in the planning model.',
         'Group encoder uses all weeks 0..51; groups reflect same users training history, not cold-start evaluation.',
         'Validation period is used to fit two calibration offsets, so validation metrics are in-sample for calibration.',
         'Rewards are regularized preference proxies, not uniquely identified economic utility.',
         'The offer schedule is replayed, not a verified intervention; no real intervention policy evaluation.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(args.output / 'local_model.npz', user_ids=user_ids, groups=groups,
                        group_weights=group_weights, user_weights=user_weights, offer_transition=offer_p,
                        coefficient_mixture_weights=mixture_weights, coefficient_mixture_means=mixture_means,
                        coefficient_mixture_covariances=covariances, simulated_purchase=sims,
                        synthetic_coefficients=synthetic_weights, synthetic_groups=synthetic_groups,
                        synthetic_population_choices=synthetic_choices)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
