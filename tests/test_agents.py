import numpy as np

from models.agents import AGENTS
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


def test_weekly_episode_updates_only_after_each_seven_day_feedback_window():
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
