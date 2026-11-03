import numpy as np
import torch

from models.nonlinear_trajectory_irl import fit_reward, fit_transitions, log_policy, make_states, policy_numpy


def test_state_and_transition_are_temporal_and_action_consistent():
    action = np.array([[0, 1, 2, 1, 0]])
    quantity = np.array([[0, 5, 1, 4, 0]])
    offers = np.array([[0, 1, 0, 1, 0]])
    states = make_states(action, quantity, offers, 2)
    assert states[0, 2] == 7  # previous action 1, high quantity and an offer
    transitions = fit_transitions(states, action, end=5)
    assert np.allclose(transitions.sum(axis=2), 1)
    for a in range(3):
        assert np.all(transitions[:, a, np.arange(12) // 4 != a] == 0)


def test_stochastic_soft_bellman_policy_is_normalized_and_uses_future():
    transition = torch.zeros((12, 3, 12), dtype=torch.float64)
    for a in range(3):
        transition[:, a, a*4] = 1
    reward = torch.zeros((12, 3), dtype=torch.float64)
    reward[4, 1] = 3
    p = log_policy(reward, transition, 3).exp().numpy()
    assert np.allclose(p.sum(axis=2), 1)
    assert p[0, 0, 1] > p[-1, 0, 1]


def test_neural_reward_can_fit_trajectory_and_produces_policy():
    rng = np.random.default_rng(41)
    actions = rng.integers(0, 3, size=(50, 6))
    quantity = rng.integers(0, 5, size=(50, 6))
    offers = rng.integers(0, 2, size=(50, 6))
    states = make_states(actions, quantity, offers, 2)
    transition = fit_transitions(states, actions, end=6)
    reward, info = fit_reward(states[:, 1:5], actions[:, 1:5], transition, nonlinear=True, seed=9)
    assert info['iterations'] > 0
    assert reward.shape == (12, 3) and reward[0, 0] == 0
    assert np.allclose(policy_numpy(reward, transition, 4).sum(-1), 1)
