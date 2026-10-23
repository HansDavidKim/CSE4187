"""Finite-horizon MaxEnt behavior model with an external binary offer record."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize


def offer_transition(offers: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if offers.ndim != 2 or offers.shape[1] < 2 or not np.isin(offers, [0, 1]).all() or alpha <= 0:
        raise ValueError('expected binary offer sequences and positive smoothing')
    counts = np.full((2, 2), alpha, dtype=float)
    np.add.at(counts, (offers[:, :-1].ravel(), offers[:, 1:].ravel()), 1)
    return counts / counts.sum(axis=1, keepdims=True)


def log_policy(weights: torch.Tensor, offer_p: torch.Tensor, horizon: int, discount: float = .9) -> torch.Tensor:
    """Return [user, time, previous purchase, offer record, purchase action]."""
    if weights.ndim != 2 or weights.shape[1] != 4 or offer_p.shape != (2, 2) or horizon < 1:
        raise ValueError('invalid parameter, transition, or horizon shape')
    if not 0 <= discount <= 1:
        raise ValueError('discount must lie in [0, 1]')
    s = torch.arange(2, dtype=weights.dtype, device=weights.device)[None, :, None]
    z = torch.arange(2, dtype=weights.dtype, device=weights.device)[None, None, :]
    w = weights
    utility = w[:, 0, None, None] + w[:, 1, None, None] * s + w[:, 2, None, None] * z + w[:, 3, None, None] * s * z
    reward = torch.stack((torch.zeros_like(utility), utility), dim=-1)
    value = torch.zeros((len(w), 2, 2), dtype=w.dtype, device=w.device)
    policies = []
    for _ in range(horizon):
        # Next state is purchase action and next offer follows an estimated exogenous record process.
        continuation = torch.einsum('ij,naj->nia', offer_p, value)[:, None, :, :]
        q = reward + discount * continuation
        policies.append(torch.log_softmax(q, dim=-1))
        value = torch.logsumexp(q, dim=-1)
    return torch.stack(policies[::-1], dim=1)


def fit_reward(states: np.ndarray, offers: np.ndarray, actions: np.ndarray, offer_p: np.ndarray,
               center: np.ndarray | None = None, penalty: float = 10., maxiter: int = 300,
               row_group: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Fit shared coefficients for each group of trajectory rows."""
    if states.ndim != 2 or states.shape != offers.shape or states.shape != actions.shape or not len(states):
        raise ValueError('expected nonempty aligned user x time arrays')
    if not all(np.isin(x, [0, 1]).all() for x in (states, offers, actions)) or penalty <= 0:
        raise ValueError('binary inputs and positive penalty required')
    if states.shape[1] > 1 and not np.array_equal(states[:, 1:], actions[:, :-1]):
        raise ValueError('state must be previous observed purchase')
    p = np.asarray(offer_p, dtype=float)
    if p.shape != (2, 2) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('invalid offer transition probabilities')
    n, horizon = states.shape
    group = np.arange(n) if row_group is None else np.asarray(row_group, dtype=int)
    if group.shape != (n,) or not np.array_equal(np.unique(group), np.arange(group.max() + 1)):
        raise ValueError('row groups must be contiguous nonnegative indices')
    n_groups = int(group.max()) + 1
    target = np.zeros((n_groups, 4)) if center is None else np.broadcast_to(center, (n_groups, 4)).copy()
    st, off, act = [torch.as_tensor(x, dtype=torch.long) for x in (states, offers, actions)]
    pt = torch.as_tensor(p, dtype=torch.float64)
    target_t = torch.as_tensor(target, dtype=torch.float64)
    gt = torch.as_tensor(group, dtype=torch.long)
    times = torch.arange(horizon)[None, :]

    def objective(flat):
        w = torch.tensor(flat.reshape(n_groups, 4), dtype=torch.float64, requires_grad=True)
        logp = log_policy(w, pt, horizon)
        loss = -logp[gt[:, None], times, st, off, act].sum() + .5 * penalty * ((w - target_t) ** 2).sum()
        loss.backward()
        return float(loss.detach()), w.grad.detach().numpy().ravel()

    result = minimize(objective, target.ravel(), jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-9})
    if not result.success:
        raise RuntimeError(f'conditional trajectory reward fit failed: {result.message}')
    return result.x.reshape(n_groups, 4), {'iterations': int(result.nit), 'objective': float(result.fun)}


def probabilities(weights: np.ndarray, offer_p: np.ndarray, horizon: int) -> np.ndarray:
    with torch.no_grad():
        return log_policy(torch.as_tensor(weights, dtype=torch.float64),
                          torch.as_tensor(offer_p, dtype=torch.float64), horizon).exp().numpy()


def sample_coefficients(mixture_weights: np.ndarray, means: np.ndarray, covariances: np.ndarray,
                        count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1 or means.ndim != 2 or means.shape[1] != 4 or covariances.shape != (len(means), 4, 4):
        raise ValueError('invalid mixture parameters')
    if mixture_weights.shape != (len(means),) or np.any(mixture_weights < 0) or not np.isclose(mixture_weights.sum(), 1):
        raise ValueError('invalid mixture weights')
    rng = np.random.default_rng(seed)
    group = rng.choice(len(means), size=count, p=mixture_weights)
    result = np.empty((count, 4))
    for i in range(len(means)):
        selected = group == i
        if selected.any():
            result[selected] = rng.multivariate_normal(means[i], covariances[i] + np.eye(4)*1e-8, size=selected.sum())
    return result, group
