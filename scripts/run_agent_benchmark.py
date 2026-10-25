"""Online learning and frozen-policy evaluation for capped coupon policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from models.agents import AGENTS
from simulate.campaign import DailyCouponEnv
from utils.paths import PROJECT_ROOT

CHECKPOINTS = (100, 500, 1_000)


def load_environment(horizon_days):
    baseline = PROJECT_ROOT / "artifacts/cosmetics-shop-persona-quadratic-irl"
    gap = json.loads((baseline / "gap_token_irl_model.json").read_text())
    uplift = json.loads((PROJECT_ROOT / "artifacts/tianchi-uplift-model/model.json").read_text())
    return DailyCouponEnv(gap, uplift, horizon_days=horizon_days)


def episode(env, policy, seed, training):
    """Make one decision per week and return the whole cooldown as feedback."""
    state = env.reset(seed)
    totals = {"reward": 0., "purchases": 0, "coupons": 0, "sessions": 0,
              "persona": env.persona}
    done = False
    while not done:
        action = policy.act(state, training=training)
        decision_state = state
        interval_reward = 0.
        for day in range(7):
            next_state, reward, done, info = env.step(action if day == 0 else 0)
            interval_reward += reward
            totals["purchases"] += info["purchase"]
            totals["coupons"] += info["coupon_sent"]
            totals["sessions"] += info["session"]
            if done:
                break
        if training:
            policy.update(decision_state, action, interval_reward, next_state, done)
        totals["reward"] += interval_reward
        state = next_state
    return totals


def rule_episode(env, seed, rule):
    state = env.reset(seed)
    totals = {"reward": 0., "purchases": 0, "coupons": 0, "sessions": 0,
              "persona": env.persona}
    done = False
    while not done:
        action = int(rule(env, state))
        for day in range(7):
            state, reward, done, info = env.step(action if day == 0 else 0)
            totals["reward"] += reward
            totals["purchases"] += info["purchase"]
            totals["coupons"] += info["coupon_sent"]
            totals["sessions"] += info["session"]
            if done:
                break
    return totals


def summarize(rows):
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    purchases = np.asarray([row["purchases"] for row in rows], dtype=float)
    coupons = np.asarray([row["coupons"] for row in rows], dtype=float)
    sessions = np.asarray([row["sessions"] for row in rows], dtype=float)
    result = {
        "mean_net_profit": float(rewards.mean()),
        "std_net_profit": float(rewards.std(ddof=1)),
        "standard_error_net_profit": float(rewards.std(ddof=1) / np.sqrt(len(rows))),
        "mean_purchases": float(purchases.mean()),
        "std_purchases": float(purchases.std(ddof=1)),
        "mean_coupons": float(coupons.mean()),
        "std_coupons": float(coupons.std(ddof=1)),
        "mean_sessions": float(sessions.mean()),
        "std_sessions": float(sessions.std(ddof=1)),
    }
    result["by_persona"] = {
        str(persona): {
            "users": len(group), "mean_net_profit": float(np.mean([row["reward"] for row in group])),
            "mean_purchases": float(np.mean([row["purchases"] for row in group])),
            "mean_coupons": float(np.mean([row["coupons"] for row in group])),
        }
        for persona in (0, 2, 3)
        if (group := [row for row in rows if row["persona"] == persona])
    }
    return result


def learning_curve(rows, oracle_rows, coupon_cost):
    cumulative_profit = np.cumsum([row["reward"] for row in rows])
    cumulative_oracle = np.cumsum([row["reward"] for row in oracle_rows])
    cumulative_coupons = np.cumsum([row["coupons"] for row in rows])
    return [{
        "users": users,
        "cumulative_net_profit": float(cumulative_profit[users - 1]),
        "campaign_spend": float(cumulative_coupons[users - 1] * coupon_cost),
        "learning_cost_vs_canonical_oracle": float(
            cumulative_oracle[users - 1] - cumulative_profit[users - 1]
        ),
    } for users in CHECKPOINTS if users <= len(rows)]


def run(train_users=1000, test_users=5000, train_weeks=13, test_weeks=52, seed=42):
    train_env = load_environment(train_weeks * 7)
    test_env = load_environment(test_weeks * 7)
    online_seed, test_seed = seed + 3_000_000, seed + 4_000_000
    oracle_online = [
        rule_episode(train_env, online_seed + user, lambda e, s: e.canonical_type == "persuadable")
        for user in range(train_users)
    ]
    never_test = [rule_episode(test_env, test_seed + user, lambda e, s: False)
                  for user in range(test_users)]
    max_test = [rule_episode(test_env, test_seed + user, lambda e, s: True)
                for user in range(test_users)]
    results = {
        "protocol": {
            "decision_interval": "7 days", "feedback_window_days": 7,
            "train_users": train_users, "train_weeks": train_weeks,
            "train_days": train_env.horizon_days, "test_users": test_users,
            "test_weeks": test_weeks, "test_days": test_env.horizon_days,
            "weekly_coupon_cap": train_env.weekly_coupon_cap,
            "purchase_value": train_env.purchase_value, "coupon_cost": train_env.coupon_cost,
            "offline_pretraining": False,
            "agent_features": list(train_env.feature_names),
            "hidden_from_agents": ["canonical_type", "canonical_probabilities", "future outcomes"],
            "neural_training_device": "mps" if torch.backends.mps.is_available() else "cpu",
            "neural_inference_device": "cpu",
            "sac": {"batch_size": 512, "online_update_interval": 8},
        },
        "test_baselines": {"Never coupon": summarize(never_test), "Max allowed": summarize(max_test)},
        "agents": {},
    }
    for index, (name, agent_class) in enumerate(AGENTS.items()):
        policy = agent_class(len(train_env.feature_names), seed=seed + index)
        started = perf_counter()
        online_rows = []
        for user in range(train_users):
            online_rows.append(episode(train_env, policy, online_seed + user, training=True))
            if name == "PPO" and (user + 1) % 32 == 0:
                policy.learn()
        if name == "PPO":
            policy.learn()
        if hasattr(policy, "sync_policy"):
            policy.sync_policy()
        training_seconds = perf_counter() - started
        test_rows = [episode(test_env, policy, test_seed + user, training=False)
                     for user in range(test_users)]
        results["agents"][name] = {
            "online": summarize(online_rows),
            "online_learning_curve": learning_curve(online_rows, oracle_online, train_env.coupon_cost),
            "test": summarize(test_rows),
            "training_wall_seconds": training_seconds,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-users", type=int, default=1000)
    parser.add_argument("--test-users", type=int, default=5000)
    parser.add_argument("--train-weeks", type=int, default=13)
    parser.add_argument("--test-weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/agent-benchmark")
    args = parser.parse_args()
    result = run(args.train_users, args.test_users, args.train_weeks, args.test_weeks, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
