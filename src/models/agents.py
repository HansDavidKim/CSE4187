"""Compact binary-action bandit and deep-RL baselines."""
from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np
import torch
from torch import nn


class UCB:
    def __init__(self, dimension, seed=0, exploration=2.0):
        self.count = np.zeros(2)
        self.total = np.zeros(2)
        self.t = 0
        self.exploration = exploration

    def act(self, state, training=True):
        if training and np.any(self.count == 0):
            return int(np.flatnonzero(self.count == 0)[0])
        mean = self.total / np.maximum(self.count, 1)
        bonus = self.exploration * np.sqrt(np.log(max(self.t, 2)) / np.maximum(self.count, 1))
        return int(np.argmax(mean + bonus)) if training else int(np.argmax(mean))

    def update(self, state, action, reward, next_state, done):
        self.t += 1
        self.count[action] += 1
        self.total[action] += reward / 1000


class ThompsonSampling(UCB):
    def __init__(self, dimension, seed=0, prior_precision=1.0):
        super().__init__(dimension)
        self.rng = np.random.default_rng(seed)
        self.precision = np.ones(2) * prior_precision

    def act(self, state, training=True):
        mean = self.total / self.precision
        if training:
            return int(np.argmax(self.rng.normal(mean, 1 / np.sqrt(self.precision))))
        return int(np.argmax(mean))

    def update(self, state, action, reward, next_state, done):
        self.t += 1
        self.precision[action] += 1
        self.total[action] += reward / 1000


class LinUCB:
    def __init__(self, dimension, seed=0, exploration=1.0):
        self.a = np.repeat(np.eye(dimension)[None], 2, axis=0)
        self.b = np.zeros((2, dimension))
        self.exploration = exploration

    def act(self, state, training=True):
        values = []
        for action in range(2):
            inverse = np.linalg.inv(self.a[action])
            mean = inverse @ self.b[action]
            bonus = self.exploration * np.sqrt(state @ inverse @ state) if training else 0
            values.append(state @ mean + bonus)
        return int(np.argmax(values))

    def update(self, state, action, reward, next_state, done):
        self.a[action] += np.outer(state, state)
        self.b[action] += state * (reward / 1000)


class LinTS(LinUCB):
    def __init__(self, dimension, seed=0, exploration=.5):
        super().__init__(dimension, seed, exploration)
        self.rng = np.random.default_rng(seed)

    def act(self, state, training=True):
        values = []
        for action in range(2):
            inverse = np.linalg.inv(self.a[action])
            mean = inverse @ self.b[action]
            theta = self.rng.multivariate_normal(mean, self.exploration ** 2 * inverse) if training else mean
            values.append(state @ theta)
        return int(np.argmax(values))


def _network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))


def _device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PPO:
    def __init__(self, dimension, seed=0, gamma=.99, learning_rate=3e-4):
        torch.manual_seed(seed)
        self.device = _device()
        self.actor = _network(dimension, 2).to(self.device)
        self.critic = _network(dimension, 1).to(self.device)
        self.actor_inference = deepcopy(self.actor).cpu()
        self.optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate
        )
        self.gamma = gamma
        self.buffer = []

    def act(self, state, training=True):
        with torch.no_grad():
            logits = self.actor_inference(torch.as_tensor(state, dtype=torch.float32))
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample() if training else logits.argmax()
        self.pending = (np.asarray(state), int(action), float(distribution.log_prob(action)))
        return int(action)

    def update(self, state, action, reward, next_state, done):
        self.buffer.append((*self.pending, reward / 1000, done))

    def learn(self):
        if not self.buffer:
            return
        states = torch.as_tensor(np.stack([row[0] for row in self.buffer]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([row[1] for row in self.buffer], device=self.device)
        old_log = torch.as_tensor([row[2] for row in self.buffer], dtype=torch.float32, device=self.device)
        rewards = [row[3] for row in self.buffer]
        dones = [row[4] for row in self.buffer]
        returns, value = [], 0.0
        for reward, done in reversed(list(zip(rewards, dones))):
            value = reward + self.gamma * value * (1 - done)
            returns.append(value)
        returns = torch.as_tensor(returns[::-1], dtype=torch.float32, device=self.device)
        advantages = returns - self.critic(states).squeeze(-1).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(4):
            distribution = torch.distributions.Categorical(logits=self.actor(states))
            ratio = torch.exp(distribution.log_prob(actions) - old_log)
            clipped = torch.clamp(ratio, .8, 1.2) * advantages
            actor_loss = -torch.minimum(ratio * advantages, clipped).mean() - .01 * distribution.entropy().mean()
            critic_loss = .5 * (self.critic(states).squeeze(-1) - returns).square().mean()
            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.optimizer.step()
        self.sync_policy()
        self.buffer.clear()

    def sync_policy(self):
        self.actor_inference.load_state_dict({key: value.detach().cpu() for key, value in self.actor.state_dict().items()})


class DiscreteSAC:
    def __init__(self, dimension, seed=0, gamma=.99, learning_rate=3e-4, alpha=.1,
                 train_interval=8, batch_size=512):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = _device()
        self.actor = _network(dimension, 2).to(self.device)
        self.actor_inference = deepcopy(self.actor).cpu()
        self.q1, self.q2 = _network(dimension, 2).to(self.device), _network(dimension, 2).to(self.device)
        self.target1, self.target2 = _network(dimension, 2).to(self.device), _network(dimension, 2).to(self.device)
        self.target1.load_state_dict(self.q1.state_dict())
        self.target2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=learning_rate)
        self.gamma, self.alpha = gamma, alpha
        self.train_interval, self.batch_size, self.steps = train_interval, batch_size, 0
        self.buffer = deque(maxlen=50_000)

    def act(self, state, training=True):
        with torch.no_grad():
            probabilities = torch.softmax(
                self.actor_inference(torch.as_tensor(state, dtype=torch.float32)), -1
            ).numpy()
        return int(self.rng.choice(2, p=probabilities)) if training else int(np.argmax(probabilities))

    def update(self, state, action, reward, next_state, done):
        self.buffer.append((np.asarray(state), action, reward / 1000, np.asarray(next_state), done))
        self.steps += 1
        if len(self.buffer) < 128 or self.steps % self.train_interval:
            return
        self._learn()
        if self.steps % 256 == 0:
            self.sync_policy()

    def sync_policy(self):
        self.actor_inference.load_state_dict({key: value.detach().cpu() for key, value in self.actor.state_dict().items()})

    def _learn(self):
        indices = self.rng.choice(len(self.buffer), min(self.batch_size, len(self.buffer)), replace=False)
        rows = [self.buffer[index] for index in indices]
        states = torch.as_tensor(np.stack([row[0] for row in rows]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([row[1] for row in rows], device=self.device).long()
        rewards = torch.as_tensor([row[2] for row in rows], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([row[3] for row in rows]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([row[4] for row in rows], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.actor(next_states)
            probabilities = torch.softmax(logits, -1)
            log_probabilities = torch.log_softmax(logits, -1)
            target_q = torch.minimum(self.target1(next_states), self.target2(next_states))
            target = rewards + self.gamma * (1 - dones) * (
                probabilities * (target_q - self.alpha * log_probabilities)
            ).sum(-1)
        q1 = self.q1(states).gather(1, actions[:, None]).squeeze(1)
        q2 = self.q2(states).gather(1, actions[:, None]).squeeze(1)
        q_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.q_optimizer.zero_grad(); q_loss.backward(); self.q_optimizer.step()
        logits = self.actor(states)
        probabilities = torch.softmax(logits, -1)
        log_probabilities = torch.log_softmax(logits, -1)
        q = torch.minimum(self.q1(states), self.q2(states)).detach()
        actor_loss = (probabilities * (self.alpha * log_probabilities - q)).sum(-1).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target1.parameters(), self.q1.parameters()):
                target_parameter.mul_(.995).add_(parameter, alpha=.005)
            for target_parameter, parameter in zip(self.target2.parameters(), self.q2.parameters()):
                target_parameter.mul_(.995).add_(parameter, alpha=.005)


AGENTS = {
    "UCB": UCB, "TS": ThompsonSampling, "LinUCB": LinUCB,
    "LinTS": LinTS, "PPO": PPO, "SAC": DiscreteSAC,
}
