"""Daily coupon-allocation environment built on the calibrated Tianchi model."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import expit, logit

from simulate.domain import sample_domain_user
from simulate.latent_session_model import (
    action_probability_offset,
    coupon_action_probabilities,
    coupon_offsets,
    lifecycle_action_probabilities,
    sample_lifecycle_step,
    sample_long_term_buyer,
    sample_home_persona,
    sample_reward_coefficients,
    sample_uplift_profile,
    update_gap_internal_state,
    _rollout_action_features,
)


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


class SessionCouponEnv:
    """Event/session simulator with weekly coupon decisions and hidden user types."""

    feature_names = DailyCouponEnv.feature_names

    def __init__(
        self, coefficient_model, action_model, gap_model, transition_model,
        uplift_model, domain_adapter, domain="cosmetics", horizon_days=42,
        weekly_coupon_cap=1, purchase_value=1000., cost_model="send_cost",
        coupon_send_cost=100., redemption_discount=.10, coupon_valid_days=42,
        max_events_per_session=64,
    ):
        if cost_model not in {"send_cost", "redemption_discount"}:
            raise ValueError("unknown coupon cost model")
        self.coefficient_model = coefficient_model
        self.action_model = action_model
        self.gap_model = gap_model
        self.transition_model = transition_model
        self.uplift_model = uplift_model
        self.domain_adapter = domain_adapter
        self.domain = domain
        self.horizon_days = int(horizon_days)
        self.weekly_coupon_cap = int(weekly_coupon_cap)
        self.purchase_value = float(purchase_value)
        self.cost_model = cost_model
        self.coupon_send_cost = float(coupon_send_cost)
        self.redemption_discount = float(redemption_discount)
        self.coupon_valid_days = int(coupon_valid_days)
        self.max_events_per_session = int(max_events_per_session)

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        sampled = sample_domain_user(self.domain_adapter, self.rng)
        self.persona = sampled["persona"]
        self.domain_component = sampled["component"]
        self.long_term_buyer = sample_long_term_buyer(
            self.transition_model, self.persona, self.rng
        )
        group = "P_LT" if self.long_term_buyer else None
        self.canonical_group = group or f"P{self.persona}"
        profile = sample_uplift_profile(
            self.uplift_model, self.persona, self.rng, group=group
        )
        self.canonical_type = profile["canonical_type"]
        self.coefficient_component, self.user_coefficients = sample_reward_coefficients(
            self.coefficient_model, self.rng, persona=self.persona
        )
        initial = np.log1p(self.gap_model["initial_gap_hours"]) / np.log1p(720.)
        self.internal_state = np.array([0., 0., 0., initial, 0., 0.])
        self.signal_history = []
        self.lifecycle = "NEW"
        self.previous_gap = None
        self.previous_previous_gap = None
        self.next_session_day = 0.
        self.day = 0
        self.sessions = []
        self.purchases = []
        self.coupons = []
        self.unused_coupons = deque()
        self.completed_sessions = 0
        self.last_session = None
        self.last_purchase = None
        return self.features()

    def _recent(self, values, days):
        return sum(value > self.day - days for value in values)

    def _eligible(self):
        return self._recent(self.coupons, 7) < self.weekly_coupon_cap

    def _receipt_ages(self, at=None):
        at = self.day if at is None else at
        return [at - sent for sent in self.coupons if 0 <= at - sent < self.coupon_valid_days]

    def features(self):
        persona = [float(self.persona == value) for value in (0, 2, 3)]
        lifecycle = [
            float(self.lifecycle == "NEW"),
            float(self.lifecycle.startswith("P")),
            float(self.lifecycle in {"CHURNING", "DISSATISFIED_CHURNING", "EXIT"}),
        ]
        since_session = self.horizon_days if self.last_session is None else self.day - self.last_session
        since_purchase = self.horizon_days if self.last_purchase is None else self.day - self.last_purchase
        rho = self.uplift_model["decay"]["rho"]
        exposure = sum(
            max(0., 1 - int(age // 7) / 5) ** rho for age in self._receipt_ages()
        ) ** .5
        weekday = 2 * np.pi * (self.day % 7) / 7
        return np.asarray([
            *persona, *lifecycle, self.day / max(self.horizon_days - 1, 1),
            np.sin(weekday), np.cos(weekday),
            min(since_session, 42) / 42, min(since_purchase, 42) / 42,
            self._recent(self.sessions, 7) / 7, self._recent(self.purchases, 7) / 7,
            self._recent(self.coupons, 7) / self.weekly_coupon_cap,
            self._recent(self.sessions, 30) / 30, self._recent(self.purchases, 30) / 30,
            self._recent(self.coupons, 30) / max(1, self.weekly_coupon_cap * 4),
            min(exposure, 3) / 3, float(self._eligible()),
        ], dtype=np.float32)

    def _policy(self, features, receipt_ages):
        probability = lifecycle_action_probabilities(
            self.action_model, self.persona, features, self.internal_state,
            self.lifecycle == "NEW", self.user_coefficients,
        )
        if self.long_term_buyer:
            probability = action_probability_offset(
                probability, 3,
                self.transition_model["long_term_buyer"]["purchase_logit_offset"],
            )
        probability = coupon_action_probabilities(
            probability, self.uplift_model, self.persona,
            self.canonical_type, receipt_ages, strength=None,
        )
        probability *= np.exp(np.asarray(self.domain_adapter["action_logit_offsets"]))
        return probability / probability.sum()

    def _session(self, at):
        receipt_ages = self._receipt_ages(at)
        typical_gap = max(float(np.expm1(self.action_model["mean"][2])), 1e-3)
        counts = np.zeros(4)
        previous_action, elapsed = -1, 0.
        for position in range(self.max_events_per_session):
            # A receipt can trigger the first conversion in a session.  It does
            # not repeatedly boost every subsequent item in the same basket.
            action_receipt_ages = receipt_ages if counts[3] == 0 else []
            probability = self._policy(
                _rollout_action_features(position, elapsed, 0., counts, previous_action),
                action_receipt_ages,
            )
            if position and self.rng.random() < probability[-1]:
                break
            event_gap = 0. if position == 0 else min(float(self.rng.exponential(typical_gap)), 30.)
            elapsed += event_gap
            probability = self._policy(
                _rollout_action_features(position, elapsed, event_gap, counts, previous_action),
                action_receipt_ages,
            )[:-1]
            probability /= probability.sum()
            previous_action = int(self.rng.choice(4, p=probability))
            counts[previous_action] += 1
        summary = {
            "view": counts[0], "cart": counts[1], "remove": counts[2],
            "purchase": counts[3], "events": counts.sum(),
            "duration_hours": elapsed / 60,
            "previous_gap_hours": self.previous_gap,
            "previous_previous_gap_hours": self.previous_previous_gap,
            "long_term_buyer": self.long_term_buyer,
            "churn_signal_history": self.signal_history[-2:],
        }
        return counts, summary

    def _reactivate(self):
        if np.isfinite(self.next_session_day):
            return
        offset = coupon_offsets(
            self.uplift_model, self.persona, self.canonical_type,
            self._receipt_ages(), strength=None,
        )["return"]
        if self.rng.random() < expit(-3.0 + offset):
            self.lifecycle = "DISSATISFIED_CHURNING"
            self.next_session_day = self.day + self.rng.uniform(1., 7.)

    def _purchase_cost(self, purchase_count, purchase_day):
        discount_cost = 0.
        for _ in range(purchase_count):
            while (
                self.unused_coupons
                and purchase_day - self.unused_coupons[0] >= self.coupon_valid_days
            ):
                self.unused_coupons.popleft()
            if self.cost_model == "redemption_discount" and self.unused_coupons:
                self.unused_coupons.popleft()
                discount_cost += self.purchase_value * self.redemption_discount
        return discount_cost

    def advance(self, action: int, days: int = 1):
        """Advance one decision window; exact for a fixed action during cooldown."""
        if action not in (0, 1):
            raise ValueError("coupon action must be zero or one")
        if days < 1:
            raise ValueError("decision window must be positive")
        sent = int(action == 1 and self._eligible())
        if sent:
            self.coupons.append(self.day)
            self.unused_coupons.append(self.day)
            self._reactivate()

        window_end = min(self.day + int(days), self.horizon_days)
        sessions = purchases = 0
        discount_cost = 0.
        gap_tokens = []
        while self.next_session_day < window_end and sessions < 8 * days:
            at = max(self.next_session_day, float(self.day))
            counts, summary = self._session(at)
            # The benchmark values one converted session as one sale.  Raw
            # purchase-event multiplicity may represent several basket lines.
            purchase_count = int(counts[3] > 0)
            sessions += 1
            purchases += purchase_count
            self.sessions.append(at)
            self.purchases.extend([at] * purchase_count)
            self.last_session = at
            if purchase_count:
                self.last_purchase = at
            discount_cost += self._purchase_cost(purchase_count, at)
            self.completed_sessions += 1
            offsets = coupon_offsets(
                self.uplift_model, self.persona, self.canonical_type,
                self._receipt_ages(at), strength=None,
            )
            gap, updated, token, destination = sample_lifecycle_step(
                self.gap_model, self.persona, self.lifecycle,
                self.internal_state, summary, self.rng,
                completed_sessions=self.completed_sessions,
                return_logit_offset=offsets["return"],
                transition_model=self.transition_model,
                transition_domain=self.domain,
            )
            signal, _ = update_gap_internal_state(
                self.internal_state, summary,
                self.gap_model["personas"][str(self.persona)]["memory"],
                self.gap_model["initial_gap_hours"],
            )
            self.signal_history.append(signal)
            self.internal_state = updated
            self.previous_previous_gap, self.previous_gap = self.previous_gap, gap
            self.lifecycle = destination
            gap_tokens.append(token)
            if gap is None:
                self.next_session_day = np.inf
                break
            gap *= float(self.domain_adapter.get("gap_time_scale", 1.))
            self.next_session_day = at + gap / 24

        revenue = self.purchase_value * purchases
        send_cost = self.coupon_send_cost * sent if self.cost_model == "send_cost" else 0.
        reward = revenue - send_cost - discount_cost
        info = {
            "coupon_sent": sent, "purchase": purchases, "session": sessions,
            "persona": self.persona, "lifecycle": self.lifecycle,
            "canonical_type": self.canonical_type,
            "canonical_group": self.canonical_group,
            "domain_component": self.domain_component,
            "revenue": revenue, "send_cost": send_cost,
            "discount_cost": discount_cost, "profit": reward,
            "gap_tokens": gap_tokens,
        }
        self.day = window_end
        done = self.day >= self.horizon_days
        state = np.zeros(len(self.feature_names), dtype=np.float32) if done else self.features()
        return state, reward, done, info

    def step(self, action: int):
        return self.advance(action, 1)

