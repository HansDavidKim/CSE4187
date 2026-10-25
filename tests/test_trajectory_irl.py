import numpy as np
import pytest
import torch

from models.trajectory_irl import fit_trajectory_reward, policy_numpy, soft_policy


def test_soft_bellman_includes_future_value():
    reward = torch.tensor([[0., 2., 0.], [0., 3., 0.], [0., 0., 0.]], dtype=torch.float64)
    p = soft_policy(reward, 3).exp().numpy()
    assert np.allclose(p.sum(axis=-1), 1)
    assert p[0, 0, 1] > p[-1, 0, 1]


def test_trajectory_fit_recovers_behavior_on_known_mdp():
    rng = np.random.default_rng(8)
    truth = np.array([[0., 1., -1.], [-1., 1., 0.], [.5, -.5, 1.]])
    policy = policy_numpy(truth, 5)
    n = 1200
    states = np.empty((n, 5), dtype=int)
    actions = np.empty_like(states)
    state = rng.integers(3, size=n)
    for t in range(5):
        states[:, t] = state
        prob = policy[t, state]
        state = (rng.random(n)[:, None] > np.cumsum(prob, axis=1)).sum(axis=1).clip(max=2)
        actions[:, t] = state
    reward, _ = fit_trajectory_reward(states, actions)
    assert np.mean(np.abs(policy_numpy(reward, 5) - policy)) < .04


def test_trajectory_fit_rejects_broken_transition():
    with pytest.raises(ValueError):
        fit_trajectory_reward(np.array([[0, 1]]), np.array([[2, 0]]))
