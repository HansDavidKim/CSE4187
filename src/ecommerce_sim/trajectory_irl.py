"""Finite-horizon maximum-causal-entropy IRL for a small behavior MDP."""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize


def soft_policy(reward: torch.Tensor, horizon: int) -> torch.Tensor:
    """s=previous action; a=current action; next state is deterministically a."""
    if reward.shape != (3, 3) or horizon < 1:
        raise ValueError('reward must be 3x3 and horizon positive')
    value = torch.zeros(3, dtype=reward.dtype, device=reward.device)
    policies = []
    for _ in range(horizon):
        q = reward + value[None, :]
        policies.append(torch.log_softmax(q, dim=1))
        value = torch.logsumexp(q, dim=1)
    return torch.stack(policies[::-1])


def fit_trajectory_reward(states: np.ndarray, actions: np.ndarray, penalty: float = 1.0) -> tuple[np.ndarray, dict]:
    if states.ndim != 2 or actions.shape != states.shape or not len(states) or penalty <= 0:
        raise ValueError('expected nonempty aligned [trajectory, time] arrays and positive penalty')
    if np.any((states < 0) | (states > 2) | (actions < 0) | (actions > 2)):
        raise ValueError('state/action outside 0..2')
    if np.any(states[:, 1:] != actions[:, :-1]):
        raise ValueError('states must transition to the previous action')
    s = torch.as_tensor(states, dtype=torch.long)
    a = torch.as_tensor(actions, dtype=torch.long)
    time = torch.arange(states.shape[1])[None, :]

    def objective(flat):
        w = torch.tensor(flat, dtype=torch.float64, requires_grad=True)
        reward = torch.cat((torch.zeros(1, dtype=w.dtype), w)).reshape(3, 3)
        logp = soft_policy(reward, states.shape[1])
        loss = -logp[time, s, a].sum() + .5 * penalty * torch.sum(w ** 2)
        loss.backward()
        return float(loss.detach()), w.grad.detach().numpy()

    result = minimize(objective, np.zeros(8), method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-11})
    if not result.success:
        raise RuntimeError(f'trajectory reward fit failed: {result.message}')
    return np.r_[0.0, result.x].reshape(3, 3), {'iterations': int(result.nit), 'objective': float(result.fun)}


def policy_numpy(reward: np.ndarray, horizon: int) -> np.ndarray:
    with torch.no_grad():
        return soft_policy(torch.as_tensor(reward, dtype=torch.float64), horizon).exp().numpy()
