import numpy as np
import torch
import pytest

from models.agents import DiscreteSAC, PPO, ProjectedTD3


def test_neural_agents_can_train_and_evaluate_39_features_on_cpu():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = np.linspace(-1, 1, 39, dtype=np.float32)
        for agent_class in (PPO, DiscreteSAC):
            agent = agent_class(39, seed=11, device="cpu")
            initial = [value.detach().clone() for value in agent.actor.parameters()]
            for _ in range(136):
                action = agent.act(state, training=True)
                agent.update(state, action, 1000. * action - 100., np.zeros(39, dtype=np.float32), True)
            if isinstance(agent, PPO):
                agent.learn()
            agent.sync_policy()
            assert agent.device.type == "cpu"
            assert any(not torch.equal(old, value) for old, value in zip(initial, agent.actor.parameters()))
            assert all(torch.isfinite(value).all() for value in agent.actor.parameters())
            assert agent.act(state, training=False) in (0, 1)
    finally:
        torch.set_num_threads(previous_threads)


def test_sac_respects_warmup_and_counts_completed_gradient_updates():
    agent = DiscreteSAC(39, device="cpu", learning_starts=16, train_interval=1, batch_size=8)
    state = np.zeros(39, dtype=np.float32)
    for _ in range(15):
        agent.update(state, 0, 10., state, False)
    assert agent.steps == 15 and agent.gradient_steps == 0
    agent.update(state, 1, 100., state, False)
    assert agent.gradient_steps == 1
    agent.update(state, 0, 10., state, True)
    assert agent.gradient_steps == 2


def test_projected_td3_delays_actor_and_target_updates_and_keeps_actions_binary():
    agent = ProjectedTD3(39, device='cpu', learning_starts=4, batch_size=4)
    state = np.linspace(-1, 1, 39, dtype=np.float32)
    actor = [p.detach().clone() for p in agent.actor.parameters()]
    targets = [p.detach().clone() for p in agent.target1.parameters()]
    for _ in range(4):
        action = agent.act(state)
        assert action in (0, 1)
        agent.update(state, action, 1000*action-100, state, False)
    assert agent.gradient_steps == 1 and agent.actor_updates == 0
    assert all(torch.equal(old, new) for old, new in zip(actor, agent.actor.parameters()))
    assert all(torch.equal(old, new) for old, new in zip(targets, agent.target1.parameters()))
    agent.update(state, 1, 1000., state, True)
    assert agent.gradient_steps == 2 and agent.actor_updates == 1
    assert any(not torch.equal(old, new) for old, new in zip(actor, agent.actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(targets, agent.target1.parameters()))
    agent.sync_policy()
    assert agent.act(state, training=False) in (0, 1)


def test_projected_td3_terminal_transition_ignores_target_value():
    state = np.zeros(39, dtype=np.float32)
    policies = [ProjectedTD3(39, device='cpu', seed=7, learning_starts=1, batch_size=1) for _ in range(2)]
    with torch.no_grad():
        for network in (policies[1].target1, policies[1].target2):
            network[-1].bias.add_(1e6)
    for policy in policies:
        policy.update(state, 1, 1000., state, True)
    for first, second in zip(policies[0].q1.parameters(), policies[1].q1.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("argument", ("learning_starts", "train_interval", "batch_size"))
def test_sac_rejects_nonpositive_training_budgets(argument):
    with pytest.raises(ValueError, match="positive integers"):
        DiscreteSAC(39, device="cpu", **{argument: 0})


@pytest.mark.parametrize("done", (False, True))
def test_ppo_bootstraps_truncated_rollouts_but_not_terminal_episodes(done):
    agent = PPO(39, device="cpu")
    with torch.no_grad():
        for value in [*agent.actor.parameters(), *agent.critic.parameters()]:
            value.zero_()
        agent.critic[-1].bias.fill_(10.)
    agent.sync_policy()
    agent.optimizer = torch.optim.SGD([*agent.actor.parameters(), *agent.critic.parameters()], lr=.01)
    state = np.zeros(39, dtype=np.float32)
    action = agent.act(state)
    agent.update(state, action, 0., state, done)
    agent.learn()
    target = 0. if done else 10. * agent.gamma
    expected = target + (10.-target)*.99**4
    assert float(agent.critic[-1].bias.detach()) == pytest.approx(expected,abs=2e-6)
