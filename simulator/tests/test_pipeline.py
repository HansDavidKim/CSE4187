import numpy as np
import pandas as pd

from simulate.pipeline import ACTIONS, EcommercePipeline, add_lags


def demo_panel(users=40, periods=8, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(users):
        state = rng.gamma(2, 1)
        for day in range(periods):
            action = int(rng.choice(len(ACTIONS), p=[0.55, 0.15, 0.15, 0.15]))
            transactions = rng.poisson(max(0.1, state + 0.4 * (action > 0)))
            spend = transactions * rng.uniform(3, 8)
            completed = int(action > 0 and rng.random() < 0.35 + 0.05 * state)
            reward = completed * rng.choice([2, 5, 10])
            rows.append({
                "user_id": user,
                "day": day,
                "action": action,
                "action_name": ACTIONS[action],
                "transactions": transactions,
                "spend": spend,
                "offer_viewed": int(action > 0),
                "offer_completed": completed,
                "reward": reward,
                **{f"action_{name}": int(ACTIONS[action] == name) for name in ACTIONS[1:]},
            })
            state = 0.6 * state + 0.4 * transactions
    return pd.DataFrame(rows)


def test_end_to_end(tmp_path):
    panel = demo_panel()
    assert len(add_lags(panel)) == 40 * 7
    pipeline = EcommercePipeline(segments=3).fit(
        panel, discovery_method="correlation-smoke", max_rows=500
    )
    synthetic = pipeline.generate(n_users=12, periods=5)
    assert len(synthetic) == 60
    assert synthetic.action_name.isin(ACTIONS).all()
    assert (synthetic[["transactions", "spend", "offer_completed", "reward"]] >= 0).all().all()
    pipeline.save(tmp_path, synthetic_users=10, periods=4)
    assert (tmp_path / "causal_dag.json").exists()
    assert (tmp_path / "user_features.csv").exists()
    assert (tmp_path / "synthetic_panel.csv").exists()

