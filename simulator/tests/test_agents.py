import numpy as np
import pickle
import pytest

from models.agents import AGENTS, LinUCB, LinTS
from scripts.run_agent_benchmark import episode, load_environment


def test_all_agents_accept_binary_transition():
    state = np.zeros(19, dtype=np.float32)
    state[-1] = 1
    for index, agent_class in enumerate(AGENTS.values()):
        agent = agent_class(len(state), seed=index)
        action = agent.act(state, training=True)
        assert action in (0, 1)
        agent.update(state, action, 100, state, True)
        if hasattr(agent, "learn"):
            agent.learn()
        assert agent.act(state, training=False) in (0, 1)


def test_weekly_episode_updates_only_after_each_seven_day_feedback_window(legacy_session_artifacts):
    class Policy:
        def __init__(self):
            self.updates = []
        def act(self, state, training=True):
            return 1
        def update(self, state, action, reward, next_state, done):
            self.updates.append((action, reward, done))

    policy = Policy()
    result = episode(load_environment(14), policy, seed=8, training=True)
    assert len(policy.updates) == 2
    assert policy.updates[-1][-1]
    assert result["coupons"] == 2


def test_linucb_rank_one_inverse_matches_direct_openblas_solution():
    rng = np.random.default_rng(42)
    agent = LinUCB(19)
    for _ in range(100):
        state = rng.normal(size=19)
        action = int(rng.integers(2))
        agent.update(state, action, rng.normal(), state, False)
    for action in range(2):
        expected = np.linalg.solve(agent.a[action], np.eye(agent.a.shape[-1]))
        np.testing.assert_allclose(agent.a_inverse[action], expected, atol=1e-10)


@pytest.mark.parametrize("agent_class", [LinUCB, LinTS])
def test_linear_policies_learn_action_intercepts_from_zero_39_feature_states(agent_class):
    state = np.zeros(39)
    agent = agent_class(39, seed=42)
    assert agent.input_dimension == 39 and agent.b.shape == (2, 40)
    for index in range(100):
        action = index % 2
        agent.update(state, action, 1000 * action, state, True)
    assert agent.act(state, training=False) == 1
    assert np.isfinite(agent.a_inverse).all()


@pytest.mark.parametrize("agent_class", [LinUCB, LinTS])
def test_linear_policies_load_old_bias_free_checkpoints_without_changing_dimensions(agent_class):
    legacy = agent_class(39, seed=42)
    del legacy.input_dimension
    legacy.a = np.repeat(np.eye(39)[None], 2, axis=0)
    legacy.a_inverse = legacy.a.copy()
    legacy.b = np.zeros((2, 39))
    legacy.b[1, 0] = 2.
    restored = pickle.loads(pickle.dumps(legacy))
    state = np.zeros(39)
    state[0] = 1
    assert restored.act(state, training=False) == 1
    restored.update(state, 1, 1000, state, True)
    assert restored.b.shape == (2, 39)
    np.testing.assert_allclose(restored.a_inverse[1], np.linalg.inv(restored.a[1]))
