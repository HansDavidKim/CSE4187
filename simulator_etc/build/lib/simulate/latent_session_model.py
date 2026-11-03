"""Small deterministic finite-horizon MaxEnt IRL and discrete return model."""
from itertools import product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def candidate_paths(n_actions, max_events=3):
    if n_actions < 2 or max_events < 1:
        raise ValueError('invalid action space or horizon')
    return [p for length in range(1, max_events + 1) for p in product(range(n_actions), repeat=length)]


def path_features(path, latent, n_actions, max_events=3):
    """Sum reward features over actions and inferred END; view is zero reference."""
    latent = np.asarray(latent, dtype=float)
    if not 1 <= len(path) <= max_events or any(a < 0 or a >= n_actions for a in path):
        raise ValueError('invalid trajectory')
    width = 3 + len(latent)
    features = np.zeros((n_actions, width))  # one row per non-view action incl END
    previous = -1
    for step, action in enumerate([*path, n_actions]):
        if action > 0:
            features[action - 1] += np.r_[1., step / max_events,
                                           float(previous == 2), latent]
        previous = action
    return features.ravel()


def fit_reward(latents, observed, n_actions, l1=.1):
    paths = candidate_paths(n_actions)
    index = {p: i for i, p in enumerate(paths)}
    latents = np.asarray(latents)
    if len(latents) != len(observed) or not len(observed) or l1 < 0:
        raise ValueError('invalid training data')
    # Features are affine in latent probabilities: construct once per candidate.
    base = np.stack([path_features(p, np.zeros(latents.shape[1]), n_actions) for p in paths])
    basis = [np.stack([path_features(p, np.eye(latents.shape[1])[j], n_actions) for p in paths]) - base
             for j in range(latents.shape[1])]
    empirical_index = np.array([index[tuple(p)] for p in observed])
    d = base.shape[1]

    def objective(v):
        w = v[:d] - v[d:]
        scores = np.broadcast_to(base @ w, (len(latents), len(paths))).copy()
        if basis:
            scores += latents @ np.stack([b @ w for b in basis])
        logz = logsumexp(scores, axis=1)
        probs = np.exp(scores - logz[:, None])
        resid = probs
        resid[np.arange(len(resid)), empirical_index] -= 1
        grad = resid.sum(axis=0) @ base
        for j in range(latents.shape[1]):
            grad += (resid.T @ latents[:, j]) @ basis[j]
        loss = float(logz.sum() - scores[np.arange(len(scores)), empirical_index].sum())
        return loss + l1*v.sum(), np.r_[grad+l1, -grad+l1]

    fit = minimize(objective, np.zeros(2*d), method='L-BFGS-B', jac=True,
                   bounds=[(0, None)]*(2*d), options={'maxiter': 1000, 'ftol': 1e-9})
    if not fit.success:
        raise RuntimeError(fit.message)
    return fit.x[:d] - fit.x[d:], {'iterations': int(fit.nit), 'objective': float(fit.fun)}


def path_probabilities(latents, reward, n_actions):
    paths = candidate_paths(n_actions)
    probs = []
    for z in np.asarray(latents):
        f = np.stack([path_features(p, z, n_actions) for p in paths])
        scores = f @ reward
        probs.append(np.exp(scores-logsumexp(scores)))
    return paths, np.asarray(probs)


def sequential_choice(paths, probs, prefix):
    """Conditional choices from exact trajectory weights (soft Q, not raw reward)."""
    if not paths or len(paths) != len(probs) or not np.isclose(np.sum(probs), 1):
        raise ValueError('invalid trajectory distribution')
    prefix = tuple(prefix)
    mass = {}
    for p, value in zip(paths, probs):
        if p[:len(prefix)] != prefix or len(p) < len(prefix):
            continue
        action = p[len(prefix)] if len(p) > len(prefix) else 'END'
        mass[action] = mass.get(action, 0.) + value
    total = sum(mass.values())
    if total <= 0:
        raise ValueError('prefix has no support')
    return {key: value/total for key, value in mass.items()}


def coupon_outcomes(p0, p1, shared_uniform):
    """Explicit monotone coupling scenario, not identified canonical labels."""
    if not (0 <= p0 <= 1 and 0 <= p1 <= 1 and 0 <= shared_uniform <= 1):
        raise ValueError('probability outside [0,1]')
    return shared_uniform < p0, shared_uniform < p1


class LatentSessionUser:
    """One simulated user; latent responsibilities held fixed after two-session prefix."""

    def __init__(self, model, prefix, rng):
        import pandas as pd
        self.model, self.rng = model, rng
        row = pd.DataFrame([prefix], columns=model['prefix_cols'])
        if row.isna().any().any():
            raise ValueError('complete two-session prefix required')
        self.latent = model['gmm'].predict_proba(model['scaler'].transform(row))[0, :2]
        self.previous_gap_hours = float(prefix['previous_gap_hours']) * 720
        self.completed_sessions = 2

    def sample_session(self):
        paths, p = path_probabilities([self.latent], self.model['reward'], len(self.model['actions']))
        chosen = paths[self.rng.choice(len(paths), p=p[0])]
        events = [self.model['actions'][a] for a in chosen]
        return events

    def finish_session(self, events):
        """Update end-of-session state and sample delay; None means >30d, not permanent exit."""
        allowed = self.model['actions']
        if not events or len(events) > self.model.get('max_session_events', 3) or any(a not in allowed for a in events):
            raise ValueError('session has unsupported length or events')
        counts = [min(events.count(a), 20)/20 for a in ('view', 'cart', 'purchase', 'remove')]
        x = np.array([*counts, min(len(events), 20)/20,
                      min(max(self.previous_gap_hours, 0), 720)/720, *self.latent])
        edges = self.model['hazard_edges_hours']
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            bins = np.zeros(len(edges)-1); bins[i] = 1
            p = self.model['hazard'].predict_proba(np.r_[bins, x][None, :])[0, 1]
            if self.rng.random() < p:
                delay = float(self.rng.uniform(lo, hi))  # assumed within-bin uniformity
                self.previous_gap_hours = delay
                self.completed_sessions += 1
                return delay
        self.completed_sessions += 1
        return None
