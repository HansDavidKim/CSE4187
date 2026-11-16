"""Small finite-horizon maximum-causal-entropy IRL with stochastic transitions."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize


def make_states(actions: np.ndarray, quantity: np.ndarray, offers: np.ndarray, threshold: float) -> np.ndarray:
    if actions.shape != quantity.shape or actions.shape != offers.shape:
        raise ValueError('state components must have identical shapes')
    states = np.zeros_like(actions, dtype=int)
    states[:, 1:] = (actions[:, :-1] * 4 + (quantity[:, :-1] > threshold).astype(int) * 2
                     + (offers[:, :-1] > 0).astype(int))
    return states


def fit_transitions(states: np.ndarray, actions: np.ndarray, end: int = 52, alpha: float = 0.5) -> np.ndarray:
    """Estimate pooled P(s'|s,a); next-state action component is always a."""
    counts = np.zeros((12, 3, 12), dtype=float)
    for a in range(3):
        counts[:, a, a*4:(a+1)*4] = alpha
    s, a, nxt = states[:, 1:end-1].ravel(), actions[:, 1:end-1].ravel(), states[:, 2:end].ravel()
    if np.any(nxt // 4 != a):
        raise ValueError('next-state previous-action component must equal action')
    np.add.at(counts, (s, a, nxt), 1)
    return counts / counts.sum(axis=2, keepdims=True)


def state_features() -> torch.Tensor:
    s = np.arange(12)
    x = np.column_stack((np.eye(3)[s // 4], (s % 4) // 2, s % 2))
    return torch.as_tensor(x, dtype=torch.float64)


def log_policy(reward: torch.Tensor, transition: torch.Tensor, horizon: int) -> torch.Tensor:
    if reward.shape != (12, 3) or transition.shape != (12, 3, 12) or horizon < 1:
        raise ValueError('invalid reward, transition, or horizon')
    value = torch.zeros(12, dtype=reward.dtype)
    result = []
    for _ in range(horizon):
        q = reward + torch.einsum('sak,k->sa', transition, value)
        result.append(torch.log_softmax(q, dim=1))
        value = torch.logsumexp(q, dim=1)
    return torch.stack(result[::-1])


def fit_reward(states: np.ndarray, actions: np.ndarray, transition: np.ndarray,
               nonlinear: bool, seed: int, penalty: float = 1.) -> tuple[np.ndarray, dict]:
    if states.ndim != 2 or states.shape != actions.shape or not len(states) or penalty <= 0:
        raise ValueError('invalid trajectory arrays or penalty')
    if np.any((states < 0) | (states >= 12) | (actions < 0) | (actions >= 3)):
        raise ValueError('invalid state/action index')
    if not np.all(states[:, 1:] // 4 == actions[:, :-1]):
        raise ValueError('observations do not match transition structure')
    torch.manual_seed(seed)
    if nonlinear:
        model = torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.Tanh(), torch.nn.Linear(12, 3)).double()
    else:
        model = torch.nn.Linear(5, 3).double()
    x = state_features()
    p = torch.as_tensor(transition, dtype=torch.float64)
    counts = np.zeros((states.shape[1], 12, 3))
    for t in range(states.shape[1]):
        np.add.at(counts[t], (states[:, t], actions[:, t]), 1)
    counts = torch.as_tensor(counts, dtype=torch.float64)
    initial = torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy()

    def objective(flat):
        torch.nn.utils.vector_to_parameters(torch.as_tensor(flat, dtype=torch.float64), model.parameters())
        model.zero_grad()
        raw = model(x)
        # A single global reference preserves state-dependent future rewards.
        reward = raw - raw[0, 0]
        policy = log_policy(reward, p, states.shape[1])
        norm = sum((v ** 2).sum() for v in model.parameters())
        loss = -(counts * policy).sum() + penalty * norm / 2
        loss.backward()
        grad = torch.nn.utils.parameters_to_vector([v.grad for v in model.parameters()]).detach().numpy().copy()
        return float(loss.detach()), grad

    result = minimize(objective, initial, jac=True, method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-8})
    if not result.success:
        raise RuntimeError(f'reward optimization failed: {result.message}')
    torch.nn.utils.vector_to_parameters(torch.as_tensor(result.x, dtype=torch.float64), model.parameters())
    with torch.no_grad():
        raw = model(x)
        reward = (raw - raw[0, 0]).numpy()
    return reward, {'iterations': int(result.nit), 'objective': float(result.fun), 'parameters': len(initial)}


def policy_numpy(reward: np.ndarray, transition: np.ndarray, horizon: int) -> np.ndarray:
    with torch.no_grad():
        return log_policy(torch.as_tensor(reward), torch.as_tensor(transition), horizon).exp().numpy()
