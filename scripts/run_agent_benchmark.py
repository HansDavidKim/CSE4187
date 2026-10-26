"""Online learning and frozen-policy evaluation for capped coupon policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from models.agents import AGENTS
from simulate.campaign import SessionCouponEnv
from utils.paths import PROJECT_ROOT

CHECKPOINTS = (100, 500, 1_000)


def load_environment(
    horizon_days, domain="cosmetics", cost_model="send_cost",
    redemption_discount=.10,
):
    baseline = PROJECT_ROOT / "artifacts/cosmetics-shop-persona-quadratic-irl"
    coefficients = json.loads((
        PROJECT_ROOT / "artifacts/cosmetics-shop-persona-maxent-irl/coefficient_gmm.json"
    ).read_text())
    action = json.loads((baseline / "lifecycle_irl_model.json").read_text())
    gap = json.loads((baseline / "gap_token_irl_model.json").read_text())
    transition = json.loads((
        PROJECT_ROOT / "artifacts/cross-domain-lifecycle/lifecycle_model.json"
    ).read_text())
    uplift = json.loads((PROJECT_ROOT / "artifacts/tianchi-uplift-model/model.json").read_text())
    adapters = json.loads((
        PROJECT_ROOT / "artifacts/domain-simulator-profiles/model.json"
    ).read_text())
    return SessionCouponEnv(
        coefficients, action, gap, transition, uplift,
        adapters["domains"][domain], domain=domain, horizon_days=horizon_days,
        cost_model=cost_model, redemption_discount=redemption_discount,
    )


def episode(env, policy, seed, training):
    """Make one decision per week and return the whole cooldown as feedback."""
    state = env.reset(seed)
    totals = {"reward": 0., "purchases": 0, "coupons": 0, "sessions": 0,
              "revenue": 0., "send_cost": 0., "discount_cost": 0.,
              "persona": env.persona}
    done = False
    while not done:
        action = policy.act(state, training=training)
        decision_state = state
        next_state, interval_reward, done, info = env.advance(action, 7)
        totals["purchases"] += info["purchase"]
        totals["coupons"] += info["coupon_sent"]
        totals["sessions"] += info["session"]
        totals["revenue"] += info.get("revenue", 0.)
        totals["send_cost"] += info.get("send_cost", 0.)
        totals["discount_cost"] += info.get("discount_cost", 0.)
        if training:
            policy.update(decision_state, action, interval_reward, next_state, done)
        totals["reward"] += interval_reward
        state = next_state
    return totals


def rule_episode(env, seed, rule):
    state = env.reset(seed)
    totals = {"reward": 0., "purchases": 0, "coupons": 0, "sessions": 0,
              "revenue": 0., "send_cost": 0., "discount_cost": 0.,
              "persona": env.persona}
    done = False
    while not done:
        action = int(rule(env, state))
        state, reward, done, info = env.advance(action, 7)
        totals["reward"] += reward
        totals["purchases"] += info["purchase"]
        totals["coupons"] += info["coupon_sent"]
        totals["sessions"] += info["session"]
        totals["revenue"] += info.get("revenue", 0.)
        totals["send_cost"] += info.get("send_cost", 0.)
        totals["discount_cost"] += info.get("discount_cost", 0.)
    return totals


def summarize(rows):
    rewards = np.asarray([row["reward"] for row in rows], dtype=float)
    purchases = np.asarray([row["purchases"] for row in rows], dtype=float)
    coupons = np.asarray([row["coupons"] for row in rows], dtype=float)
    sessions = np.asarray([row["sessions"] for row in rows], dtype=float)
    revenue = np.asarray([row["revenue"] for row in rows], dtype=float)
    send_cost = np.asarray([row["send_cost"] for row in rows], dtype=float)
    discount_cost = np.asarray([row["discount_cost"] for row in rows], dtype=float)
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
        "mean_gross_revenue": float(revenue.mean()),
        "mean_send_cost": float(send_cost.mean()),
        "mean_discount_cost": float(discount_cost.mean()),
        "mean_coupon_cost_per_purchase": float(
            (send_cost.sum() + discount_cost.sum()) / max(purchases.sum(), 1)
        ),
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


def learning_curve(rows, oracle_rows):
    cumulative_profit = np.cumsum([row["reward"] for row in rows])
    cumulative_oracle = np.cumsum([row["reward"] for row in oracle_rows])
    cumulative_cost = np.cumsum([
        row["send_cost"] + row["discount_cost"] for row in rows
    ])
    return [{
        "users": users,
        "cumulative_net_profit": float(cumulative_profit[users - 1]),
        "campaign_spend": float(cumulative_cost[users - 1]),
        "learning_cost_vs_canonical_oracle": float(
            cumulative_oracle[users - 1] - cumulative_profit[users - 1]
        ),
    } for users in CHECKPOINTS if users <= len(rows)]


def run(
    train_users=1000, test_users=5000, train_weeks=13, test_weeks=52,
    seed=42, domain="cosmetics", cost_model="send_cost",
    redemption_discount=.10,
):
    train_env = load_environment(
        train_weeks * 7, domain, cost_model, redemption_discount
    )
    test_env = load_environment(
        test_weeks * 7, domain, cost_model, redemption_discount
    )
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
            "purchase_value": train_env.purchase_value,
            "domain": domain, "cost_model": cost_model,
            "coupon_send_cost": train_env.coupon_send_cost,
            "redemption_discount": train_env.redemption_discount,
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
            "online_learning_curve": learning_curve(online_rows, oracle_online),
            "test": summarize(test_rows),
            "training_wall_seconds": training_seconds,
        }
    return results


def generate_synthetic_dataset(
    domain, users=1000, weeks=13, seed=42,
    output=PROJECT_ROOT / "artifacts/synthetic-benchmark-datasets",
):
    """Generate one inspectable randomized-policy panel per domain."""
    env = load_environment(weeks * 7, domain, "redemption_discount")
    rows = []
    for user in range(users):
        rng = np.random.default_rng(np.random.SeedSequence([seed, user]))
        state = env.reset(seed + 5_000_000 + user)
        done = False
        week = 0
        while not done:
            action = int(rng.random() < .5)
            values = {
                "sessions": 0, "purchases": 0, "coupons": 0,
                "gross_revenue": 0., "redemption_discount_cost": 0.,
            }
            state, _, done, info = env.advance(action, 7)
            values["sessions"] += info["session"]
            values["purchases"] += info["purchase"]
            values["coupons"] += info["coupon_sent"]
            values["gross_revenue"] += info["revenue"]
            values["redemption_discount_cost"] += info["discount_cost"]
            gap_counts = {
                f"gap_{token.lower()}": info["gap_tokens"].count(token)
                for token in env.gap_model["gap_tokens"]
            }
            rows.append({
                "domain": domain, "user_id": user, "week": week,
                "coupon_action": action, "home_persona": env.persona,
                "canonical_type": env.canonical_type,
                "canonical_group": env.canonical_group,
                "coefficient_gmm_component": env.coefficient_component,
                "domain_gmm_component": env.domain_component,
                **values, **gap_counts,
                "send_cost_profit": values["gross_revenue"] - 100 * values["coupons"],
                "redemption_discount_profit": (
                    values["gross_revenue"] - values["redemption_discount_cost"]
                ),
            })
            week += 1
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{domain}.parquet"
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    (output / f"{domain}.json").write_text(json.dumps({
        "domain": domain, "users": users, "weeks": weeks, "rows": len(frame),
        "seed": seed, "decision_interval": "7 days",
        "random_coupon_probability": .5,
        "purchase_value": env.purchase_value,
        "send_cost": 100., "redemption_discount": env.redemption_discount,
        "gap_tokens": env.gap_model["gap_tokens"],
        "columns": list(frame.columns),
        "note": "canonical types and coupon effects are simulator assumptions, not observed labels",
    }, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-users", type=int, default=1000)
    parser.add_argument("--test-users", type=int, default=5000)
    parser.add_argument("--train-weeks", type=int, default=13)
    parser.add_argument("--test-weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--domains", nargs="+", default=["cosmetics", "electronics", "tianchi"],
        choices=["cosmetics", "electronics", "tianchi"],
    )
    parser.add_argument(
        "--cost-models", nargs="+", default=["send_cost", "redemption_discount"],
        choices=["send_cost", "redemption_discount"],
    )
    parser.add_argument("--redemption-discount", type=float, default=.10)
    parser.add_argument("--synthetic-users", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/agent-benchmark")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic = {}
    if args.synthetic_users > 0:
        synthetic = {
            domain: str(generate_synthetic_dataset(
                domain, args.synthetic_users, args.train_weeks, args.seed
            )) for domain in args.domains
        }
    results = {"synthetic_datasets": synthetic, "benchmarks": {}}
    for domain in args.domains:
        for cost_model in args.cost_models:
            key = f"{domain}:{cost_model}"
            result = run(
                args.train_users, args.test_users, args.train_weeks, args.test_weeks,
                args.seed, domain, cost_model, args.redemption_discount,
            )
            results["benchmarks"][key] = result
            (args.output / f"{domain}_{cost_model}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
