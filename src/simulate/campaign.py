"""Daily coupon-allocation environment built on the calibrated Tianchi model."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import expit, logit

from simulate.latent_session_model import coupon_offsets, sample_home_persona, sample_uplift_profile


class DailyCouponEnv:
    """One user over 42 daily decisions; canonical response type stays hidden."""

    feature_names = (
        "persona_0", "persona_2", "persona_3", "lifecycle_new",
        "lifecycle_mature", "lifecycle_churning", "day_fraction",
        "weekday_sin", "weekday_cos", "days_since_session", "days_since_purchase",
        "sessions_7d", "purchases_7d", "coupons_7d", "sessions_30d",
        "purchases_30d", "coupons_30d", "coupon_exposure", "coupon_eligible",
    )

    def __init__(
        self, gap_model, uplift_model, horizon_days=42, weekly_coupon_cap=1,
        purchase_value=1000.0, coupon_cost=100.0,
    ):
        if horizon_days < 1 or weekly_coupon_cap < 1:
            raise ValueError("horizon and coupon cap must be positive")
        self.gap_model = gap_model
        self.uplift_model = uplift_model
        self.horizon_days = int(horizon_days)
        self.weekly_coupon_cap = int(weekly_coupon_cap)
        self.purchase_value = float(purchase_value)
        self.coupon_cost = float(coupon_cost)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.persona = sample_home_persona(self.gap_model, self.rng)
        profile = sample_uplift_profile(self.uplift_model, self.persona, self.rng)
        self.canonical_type = profile["canonical_type"]
        self.day = 0
        self.sessions = []
        self.purchases = []
        self.coupons = []
        self.completed_sessions = 0
        self.last_session = None
        self.last_purchase = None
        # Common random numbers make policy comparisons less noisy.
        self.session_uniforms = self.rng.random(self.horizon_days)
        self.purchase_uniforms = self.rng.random(self.horizon_days)
        return self.features()

    def _recent(self, values, days):
        return sum(day > self.day - days for day in values)

    def _receipt_ages(self):
        return [self.day - sent for sent in self.coupons if self.day - sent < 42]

    def _eligible(self):
        return self._recent(self.coupons, 7) < self.weekly_coupon_cap

    def _lifecycle(self):
        if self.completed_sessions < 2:
            return "NEW"
        if self.last_session is None or self.day - self.last_session >= 7:
            return "CHURNING"
        return "MATURE"

    def features(self):
        persona = [float(self.persona == value) for value in (0, 2, 3)]
        lifecycle = self._lifecycle()
        lifecycle_one_hot = [float(lifecycle == value) for value in ("NEW", "MATURE", "CHURNING")]
        since_session = self.horizon_days if self.last_session is None else self.day - self.last_session
        since_purchase = self.horizon_days if self.last_purchase is None else self.day - self.last_purchase
        ages = self._receipt_ages()
        rho = self.uplift_model["decay"]["rho"]
        exposure = sum(max(0.0, 1 - int(age // 7) / 5) ** rho for age in ages) ** .5
        weekday = 2 * np.pi * (self.day % 7) / 7
        return np.asarray([
            *persona, *lifecycle_one_hot, self.day / max(self.horizon_days - 1, 1),
            np.sin(weekday), np.cos(weekday),
            min(since_session, 42) / 42, min(since_purchase, 42) / 42,
            self._recent(self.sessions, 7) / 7, self._recent(self.purchases, 7) / 7,
            self._recent(self.coupons, 7) / self.weekly_coupon_cap,
            self._recent(self.sessions, 30) / 30, self._recent(self.purchases, 30) / 30,
            self._recent(self.coupons, 30) / max(1, self.weekly_coupon_cap * 4),
            min(exposure, 3) / 3, float(self._eligible()),
        ], dtype=np.float32)

    @staticmethod
    def _daily_probability(period_probability, days):
        return 1 - (1 - np.clip(period_probability, 1e-6, 1 - 1e-6)) ** (1 / days)

    def step(self, action: int):
        if action not in (0, 1):
            raise ValueError("coupon action must be zero or one")
        sent = int(action == 1 and self._eligible())
        if sent:
            self.coupons.append(self.day)
        specification = self.uplift_model["personas"][str(self.persona)]
        offsets = coupon_offsets(
            self.uplift_model, self.persona, self.canonical_type,
            self._receipt_ages(), strength=None,
        )
        session_probability = self._daily_probability(specification["return_7d_control"], 7)
        session_probability = float(expit(logit(session_probability) + offsets["return"]))
        session = self.session_uniforms[self.day] < session_probability
        purchase = False
        if session:
            self.sessions.append(self.day)
            self.completed_sessions += 1
            self.last_session = self.day
            daily_purchase = self._daily_probability(specification["purchase_14d_control"], 14)
            conditional = min(daily_purchase / max(session_probability, 1e-6), .95)
            conditional = float(expit(logit(np.clip(conditional, 1e-6, 1 - 1e-6)) + offsets["purchase"]))
            purchase = self.purchase_uniforms[self.day] < conditional
            if purchase:
                self.purchases.append(self.day)
                self.last_purchase = self.day
        reward = self.purchase_value * int(purchase) - self.coupon_cost * sent
        info = {
            "coupon_sent": sent, "purchase": int(purchase), "session": int(session),
            "persona": self.persona, "lifecycle": self._lifecycle(),
        }
        self.day += 1
        done = self.day >= self.horizon_days
        return (np.zeros(len(self.feature_names), dtype=np.float32) if done else self.features()), reward, done, info

