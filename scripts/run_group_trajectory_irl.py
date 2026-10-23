#!/usr/bin/env python3
"""GRU masked sequence representation, GMM segments and finite-horizon MaxEnt IRL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from ecommerce_sim.preprocessing.nts_adapter import load_sequences
from ecommerce_sim.models.trajectory_irl import fit_trajectory_reward, policy_numpy
from ecommerce_sim.models.user_choice import state_spec, weekly_choices


class MaskedGRU(torch.nn.Module):
    def __init__(self, features, hidden=16):
        super().__init__()
        self.gru = torch.nn.GRU(features * 2, hidden, batch_first=True, bidirectional=True)
        self.decoder = torch.nn.Linear(hidden * 2, features)

    def forward(self, values, mask):
        x = torch.cat((values.masked_fill(mask, 0), mask.to(values.dtype)), dim=-1)
        hidden, _ = self.gru(x)
        return self.decoder(hidden), hidden.mean(dim=1)


def train_encoder(data, seed, epochs):
    torch.manual_seed(seed)
    x = torch.tensor(data, dtype=torch.float32)
    model = MaskedGRU(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for indices in order.split(128):
            batch = x[indices]
            mask = torch.rand(batch.shape, generator=generator) < .15
            if not mask.any():
                continue
            prediction, _ = model(batch, mask)
            loss = ((prediction - batch)[mask] ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(indices)
        if epoch in (0, epochs - 1):
            print(f'GRU epoch {epoch+1}/{epochs}: masked training MSE={total/len(x):.4f}', flush=True)
    model.eval()
    with torch.no_grad():
        _, embeddings = model(x, torch.zeros_like(x, dtype=torch.bool))
    return model, embeddings.numpy()


def sequence_metrics(a):
    before = a[..., :-1] > 0
    after = a[..., 1:] > 0
    return {'rates': (np.bincount(a.ravel(), minlength=3) / a.size).tolist(),
            'switch_rate': float((a[..., 1:] != a[..., :-1]).mean()),
            'purchase_after_purchase': float(after[before].mean()) if before.any() else None}


def evaluate(actions, first_state, groups, policies):
    previous = np.column_stack((first_state, actions[:, :-1]))
    prediction = policies[groups[:, None], np.arange(actions.shape[1])[None, :], previous]
    chosen = np.take_along_axis(prediction, actions[..., None], axis=-1)[..., 0]
    return {'log_loss': float(-np.log(np.clip(chosen, 1e-12, 1)).mean()),
            'brier': float(((prediction - np.eye(3)[actions]) ** 2).sum(axis=-1).mean()),
            'accuracy': float((prediction.argmax(axis=-1) == actions).mean())}


def rollout(first, group, policies, n_rollouts, seed):
    rng = np.random.default_rng(seed)
    result = np.zeros((n_rollouts, len(first), policies.shape[1]), dtype=int)
    for k in range(n_rollouts):
        state = first.copy()
        for t in range(policies.shape[1]):
            p = policies[group, t, state]
            state = np.minimum((rng.random(len(first))[:, None] > np.cumsum(p, axis=1)).sum(axis=1), 2)
            result[k, :, t] = state
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequences', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_sequences.npz'))
    parser.add_argument('--panel', type=Path, default=Path('artifacts/amexpert-panel-observed/customer_week_panel.csv.gz'))
    parser.add_argument('--dag', type=Path, default=Path('artifacts/amexpert-nts-train-2026/nts_dag.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/amexpert-group-trajectory-irl'))
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--rollouts', type=int, default=10)
    args = parser.parse_args()
    if args.groups < 1 or args.epochs < 1 or args.rollouts < 1:
        parser.error('groups, epochs and rollouts must be positive')
    torch.set_num_threads(1)
    sequences, features = load_sequences(args.sequences)
    values = np.stack(sequences)
    if values.shape[1] != 78:
        raise ValueError('pilot requires 78 weeks per customer')
    panel = pd.read_csv(args.panel).sort_values(['customer_id', 'week'])
    grouped = panel.groupby('customer_id', sort=True)
    user_ids = grouped.size().index.to_numpy()
    if len(user_ids) != len(values) or not np.all(grouped.size().to_numpy() == 78) or not np.allclose(panel[features].to_numpy(), values.reshape(-1, len(features))):
        raise ValueError('panel does not match sequence archive')
    dag = json.loads(args.dag.read_text())
    spec = state_spec(dag)
    # Coupon discount is an observed label ingredient, not a graph-derived pre-action state.
    encoder_features = sorted({name for name, _ in spec} | {'coupon_discount'})
    raw_train = values[:, :52, [features.index(name) for name in encoder_features]]
    logged = np.log1p(np.maximum(raw_train, 0))
    mean, scale = logged.mean(axis=(0, 1)), logged.std(axis=(0, 1))
    scale[scale == 0] = 1
    encoder, embeddings = train_encoder((logged - mean) / scale, args.seed, args.epochs)
    emb_scaler = StandardScaler().fit(embeddings)
    scaled = emb_scaler.transform(embeddings)
    gmm = GaussianMixture(n_components=args.groups, reg_covar=1e-4, random_state=args.seed, n_init=3).fit(scaled)
    groups = gmm.predict(scaled)
    actions = weekly_choices(values, features)
    # Four episodes cover the training horizon; week 39 is counted twice.
    starts = (1, 14, 27, 39)
    train_actions = np.stack([actions[:, start:start+13] for start in starts], axis=1)
    train_states = np.stack([actions[:, start-1:start+12] for start in starts], axis=1)
    rewards, diagnostics = [], []
    for group in range(args.groups):
        selected = groups == group
        if not selected.any():
            raise ValueError('empty estimated group; choose fewer groups')
        r, diag = fit_trajectory_reward(train_states[selected].reshape(-1, 13), train_actions[selected].reshape(-1, 13))
        rewards.append(r)
        diagnostics.append({'group': group, 'users': int(selected.sum()), **diag})
    pooled, pooled_diag = fit_trajectory_reward(train_states.reshape(-1, 13), train_actions.reshape(-1, 13))
    policies = np.stack([policy_numpy(r, 13) for r in rewards])
    pooled_policy = policy_numpy(pooled, 13)[None, ...]
    y, initial = actions[:, 65:78], actions[:, 64]
    simulations = rollout(initial, groups, policies, args.rollouts, args.seed)
    observed_user = np.stack([(y == a).mean(axis=1) for a in range(3)], axis=-1)
    generated_user = np.stack([(simulations == a).mean(axis=(0, 2)) for a in range(3)], axis=-1)
    summary = {'method': 'bidirectional masked GRU + GMM + group finite-horizon maximum causal entropy IRL',
        'source_dag': str(args.dag), 'encoder_features': encoder_features, 'group_count': args.groups,
        'seed': args.seed, 'gru_epochs': args.epochs, 'train_weeks_encoder': [0, 51],
        'irl_training_episodes': [[1, 13], [14, 26], [27, 39], [39, 51]], 'validation_weeks': [52, 64],
        'test_weeks': [65, 77], 'horizon': 13, 'transition': 'next_state = action',
        'actions': ['no_purchase', 'purchase_without_recorded_coupon_discount', 'purchase_with_recorded_coupon_discount'],
        'group_fit': diagnostics, 'pooled_fit': pooled_diag,
        'validation_teacher_forced_group': evaluate(actions[:, 52:65], actions[:, 51], groups, policies),
        'test_teacher_forced_group': evaluate(y, initial, groups, policies),
        'test_teacher_forced_pooled': evaluate(y, initial, np.zeros(len(y), dtype=int), pooled_policy),
        'test_observed_sequence': sequence_metrics(y), 'test_generated_sequence': sequence_metrics(simulations),
        'test_by_group': [{'group': i, 'observed': sequence_metrics(y[groups == i]),
                           'generated': sequence_metrics(simulations[:, groups == i])}
                          for i in range(args.groups)],
        'per_user_rate_mae': np.abs(observed_user-generated_user).mean(axis=0).tolist(),
        'rollouts': args.rollouts,
        'limitations': ['MDP state is previous behavior category, not full NTS causal state.',
                        'Only group-specific rewards, no separately fitted user-specific trajectory reward.',
                        'GRU encoder is masked reconstruction, not a Transformer or pretrained BERT.',
                        'Rewards are non-identifiable up to transformations; fixed baseline and regularization select one solution.',
                        'No item choice set, observed exposures, or causal intervention identification.',
                        'Episodes overlap at week 39 and same-user episodes are dependent; no group stability or cold-start test.',
                        'NTS graph influences encoder inputs only, not structural transition constraints.']}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savez_compressed(args.output / 'local_groups_rewards.npz', user_ids=user_ids, groups=groups,
        responsibilities=gmm.predict_proba(scaled), embeddings=embeddings, rewards=np.asarray(rewards),
        policies=policies, pooled_reward=pooled, encoder_mean=mean, encoder_scale=scale,
        encoder_features=encoder_features, embedding_mean=emb_scaler.mean_, embedding_scale=emb_scaler.scale_,
        gmm_weights=gmm.weights_, gmm_means=gmm.means_, gmm_covariances=gmm.covariances_)
    torch.save(encoder.state_dict(), args.output / 'local_masked_gru.pt')
    np.savez_compressed(args.output / 'local_simulated_trajectories.npz', user_ids=user_ids, choices=simulations)
    print(json.dumps({'group_sizes': [d['users'] for d in diagnostics],
        'group_model': summary['test_teacher_forced_group'], 'pooled_model': summary['test_teacher_forced_pooled'],
        'observed': summary['test_observed_sequence'], 'generated': summary['test_generated_sequence'],
        'per_user_rate_mae': summary['per_user_rate_mae']}, indent=2))


if __name__ == '__main__':
    main()
