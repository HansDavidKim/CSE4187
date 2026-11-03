import json
from pathlib import Path

import numpy as np

from simulate.campaign import DailyCouponEnv


ROOT = Path(__file__).parents[1]


def models():
    gap = json.loads((ROOT / "artifacts/cosmetics-shop-persona-quadratic-irl/gap_token_irl_model.json").read_text())
    uplift = json.loads((ROOT / "artifacts/tianchi-uplift-model/model.json").read_text())
    return gap, uplift


def test_daily_environment_enforces_weekly_cap_and_hides_canonical_type():
    gap, uplift = models()
    env = DailyCouponEnv(gap, uplift, horizon_days=10, weekly_coupon_cap=3)
    state = env.reset(7)
    assert state.shape == (len(env.feature_names),)
    assert not any("canonical" in name for name in env.feature_names)
    sent = 0
    for _ in range(7):
        state, reward, done, info = env.step(1)
        sent += info["coupon_sent"]
        assert np.isfinite(reward)
    assert sent == 3 and not done
    assert env.step(1)[3]["coupon_sent"] == 1


def test_coupon_reward_is_charged_only_when_message_is_sent():
    gap, uplift = models()
    env = DailyCouponEnv(gap, uplift, horizon_days=2, weekly_coupon_cap=1)
    env.reset(11)
    _, first, _, first_info = env.step(1)
    _, second, done, second_info = env.step(1)
    assert first_info["coupon_sent"] == 1
    assert second_info["coupon_sent"] == 0
    assert first % 100 == 0 and second % 100 == 0 and done
