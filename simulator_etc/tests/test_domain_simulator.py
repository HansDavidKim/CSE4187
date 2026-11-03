import json

import numpy as np

from scripts.run_agent_benchmark import load_environment
from simulate.domain import sample_domain_user
from utils.paths import PROJECT_ROOT


def test_domain_gmm_samples_supported_personas():
    model = json.loads((
        PROJECT_ROOT / "artifacts/domain-simulator-profiles/model.json"
    ).read_text())
    rng = np.random.default_rng(9)
    for adapter in model["domains"].values():
        rows = [sample_domain_user(adapter, rng) for _ in range(100)]
        assert {row["persona"] for row in rows} <= {0, 2, 3}
        assert all(len(row["domain_latent"]) == len(adapter["features"]) for row in rows)


def test_cost_models_preserve_behavior_and_change_only_accounting():
    send = load_environment(14, "cosmetics", "send_cost")
    discount = load_environment(14, "cosmetics", "redemption_discount")
    send.reset(17)
    discount.reset(17)
    send_totals = {"purchase": 0, "session": 0, "reward": 0.}
    discount_totals = {"purchase": 0, "session": 0, "reward": 0.}
    for action in (1, 1):
        _, reward_a, _, info_a = send.advance(action, 7)
        _, reward_b, _, info_b = discount.advance(action, 7)
        for key in ("purchase", "session"):
            send_totals[key] += info_a[key]
            discount_totals[key] += info_b[key]
        send_totals["reward"] += reward_a
        discount_totals["reward"] += reward_b
        assert info_a["gap_tokens"] == info_b["gap_tokens"]
    assert send_totals["purchase"] == discount_totals["purchase"]
    assert send_totals["session"] == discount_totals["session"]
    assert send_totals["reward"] <= 1000 * send_totals["purchase"]
    assert discount_totals["reward"] <= 1000 * discount_totals["purchase"]

