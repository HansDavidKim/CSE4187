"""Finite-horizon causal-entropy reward fitting with variable-length sessions."""
import numpy as np
import torch
from ecommerce_sim.simulate.latent_session_model import LatentSessionUser


def policy_log_probs(weights, latent, n_actions, horizon):
    """Soft Bellman recursion; state is event count and previous purchase flag."""
    if weights.shape != (n_actions, 4 + latent.shape[1]) or horizon < 2:
        raise ValueError('invalid weights or horizon')
    n = latent.shape[0]
    values = [torch.zeros((n, 2), dtype=latent.dtype, device=latent.device)] * (horizon + 1)
    policies = [None] * horizon
    for t in range(horizon - 1, -1, -1):
        state_policies, state_values = [], []
        for previous_purchase in (0, 1):
            fraction = t / horizon
            f = torch.cat([torch.ones((n, 1), dtype=latent.dtype, device=latent.device),
                           torch.full((n, 1), fraction, dtype=latent.dtype, device=latent.device),
                           torch.full((n, 1), fraction*fraction, dtype=latent.dtype, device=latent.device),
                           torch.full((n, 1), previous_purchase, dtype=latent.dtype, device=latent.device), latent], dim=1)
            reward = torch.cat([torch.zeros((n, 1), dtype=latent.dtype, device=latent.device), f @ weights.T], dim=1)
            continuation = torch.stack([values[t+1][:, int(action == 2)] for action in range(n_actions)], dim=1)
            q = torch.cat([reward[:, :n_actions] + continuation, reward[:, n_actions:]], dim=1)
            if t == 0:
                q = torch.cat([q[:, :n_actions], torch.full((n, 1), -torch.inf, dtype=q.dtype, device=q.device)], dim=1)
            state_policies.append(torch.log_softmax(q, dim=1))
            state_values.append(torch.logsumexp(q, dim=1))
        policies[t] = torch.stack(state_policies, dim=1)
        values[t] = torch.stack(state_values, dim=1)
    return torch.stack(policies, dim=1)


def encode_sessions(paths, n_actions, horizon):
    targets = np.full((len(paths), horizon), -1, dtype=int)
    previous = np.zeros((len(paths), horizon), dtype=int)
    for i, seq in enumerate(paths):
        if not seq or any(a < 0 or a >= n_actions for a in seq):
            raise ValueError('invalid observed path')
        length = min(len(seq), horizon)
        targets[i, :length] = seq[:length]
        for t in range(1, length):
            previous[i, t] = int(seq[t-1] == 2)
        if len(seq) < horizon:
            targets[i, length] = n_actions
            previous[i, length] = int(seq[-1] == 2)
    return targets, previous


def fit_model(latents, paths, n_actions, horizon=30, l1=0.0001):
    z = torch.as_tensor(np.asarray(latents), dtype=torch.float64)
    if len(z) != len(paths) or not len(paths):
        raise ValueError('nonempty aligned paths required')
    targets, previous = encode_sessions(paths, n_actions, horizon)
    row, time = np.where(targets >= 0)
    target = torch.as_tensor(targets[row, time])
    prev = torch.as_tensor(previous[row, time])
    weights = torch.nn.Parameter(torch.zeros((n_actions, 4 + z.shape[1]), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([weights], lr=.8, max_iter=150, tolerance_grad=1e-7,
                                  tolerance_change=1e-10, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        policy = policy_log_probs(weights, z, n_actions, horizon)
        loss = -policy[row, time, prev, target].mean() + l1*torch.sqrt(weights*weights + 1e-10).sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        objective = float(-policy_log_probs(weights, z, n_actions, horizon)[row, time, prev, target].mean())
    return weights.detach().numpy(), {'train_nll_per_observed_decision': objective,
                                      'train_decisions': len(row)}


def evaluate(latents, paths, weights, n_actions, horizon=30, draws=200, seed=42):
    z = torch.as_tensor(np.asarray(latents), dtype=torch.float64)
    targets, previous = encode_sessions(paths, n_actions, horizon)
    row, time = np.where(targets >= 0)
    with torch.no_grad():
        logp = policy_log_probs(torch.as_tensor(weights), z, n_actions, horizon).numpy()
    nll = float(-logp[row, time, previous[row, time], targets[row, time]].mean())
    rng = np.random.default_rng(seed)
    generated, capped = [], 0
    for i in range(len(paths)):
        for _ in range(draws):
            path, prev = [], 0
            for t in range(horizon):
                probs = np.exp(logp[i, t, prev])
                a = int(rng.choice(n_actions+1, p=probs/probs.sum()))
                if a == n_actions:
                    break
                path.append(a)
                prev = int(a == 2)
            else:
                capped += 1
            generated.append(path)
    def distribution(seqs):
        lengths = np.asarray([len(s) for s in seqs])
        counts = np.bincount([a for s in seqs for a in s], minlength=n_actions)
        return {'length_bins_1_2_3_4to10_11to29_30plus': [int((lengths == 1).sum()), int((lengths == 2).sum()),
                 int((lengths == 3).sum()), int(((lengths >= 4) & (lengths <= 10)).sum()),
                 int(((lengths >= 11) & (lengths < 30)).sum()), int((lengths >= 30).sum())],
                 'events': int(counts.sum()), 'action_counts': counts.tolist(),
                 'length_median': float(np.median(lengths))}
    return {'nll_per_observed_decision': nll, 'observed': distribution(paths),
            'generated': distribution(generated), 'draws_per_user': draws,
            'generated_capped': capped}


class ExtendedSessionUser(LatentSessionUser):
    def sample_session(self):
        h = self.model['max_session_events']
        actions = self.model['actions']
        with torch.no_grad():
            lp = policy_log_probs(torch.as_tensor(self.model['extended_reward']),
                                  torch.as_tensor(self.latent[None, :]), len(actions), h).numpy()[0]
        events, prev = [], 0
        for t in range(h):
            p = np.exp(lp[t, prev]); p /= p.sum()
            action = int(self.rng.choice(len(actions) + 1, p=p))
            if action == len(actions):
                return {'events': events, 'ended': True}
            events.append(actions[action])
            prev = int(action == 2)
        return {'events': events, 'ended': False}
