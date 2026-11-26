"""Point-in-time KIT39 covariates from the consumed Cosmetics source prefix.

Cosmetics has no signup, order ID, quantity, coupon or review instrumentation.
Checkout features are explicitly labelled proxies; absolute cart and unsupported
channels stay unknown. No synthetic reviews, future catalog, labels or personas
are used. This module does not alter the existing public feature extractors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from ..features import EXPANDED_FEATURES
from ..time_spec import feature_window_days


ACTIONS = ("view", "cart", "remove", "purchase", "end")
UNOBSERVED_CHANNEL_FEATURES = (
    "coupon_uses_30d", "coupon_redemption_recency_days", "coupon_discount_amount_30d",
    "review_count_30d", "mean_review_rating_30d", "review_last_order_mean_rating",
    "review_recency_days", "review_non5_share_30d",
)
UNKNOWN_INITIAL_CART_FEATURES = ("cart_qty_current", "cart_sku_current", "cart_gross_value_current")
CHECKOUT_PROXY_FEATURES = (
    "purchase_count_30d", "net_spend_30d", "mean_net_order_value_30d", "orders_per_session_30d",
    "mean_purchase_gap_days", "purchased_qty_30d", "mean_basket_qty_30d", "orders_delta_30d",
    "net_spend_delta_30d", "mean_net_order_value_delta_30d", "purchased_qty_delta_30d",
)
_DELTA_FEATURES = (
    "orders_delta_30d", "net_spend_delta_30d", "mean_net_order_value_delta_30d",
    "unique_purchase_products_delta_30d", "purchased_qty_delta_30d",
)
_GAP_FEATURES = ("intervisit_start_gap_mean_days_90d", "intervisit_start_gap_cv_90d")


def _utc(value):
    clock = pd.Timestamp(value)
    if pd.isna(clock) or clock.tzinfo is None:
        raise ValueError("source feature clocks must be known and timezone-aware")
    return clock.tz_convert("UTC")


def _divide(a, b):
    return float(a / b) if b else math.nan


def _mean(values):
    return float(np.mean(values)) if values else math.nan


def _days(later, earlier):
    return (later - earlier).total_seconds() / 86400


def _amount(orders):
    return math.nan if any(order["amount"] is None for order in orders) else float(sum(order["amount"] for order in orders))


@dataclass
class _Prefix:
    observation_start: pd.Timestamp
    observation_end: pd.Timestamp
    windows: tuple[int, int, int]
    checkout_proxy: bool
    events: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    starts: list = field(default_factory=list)
    completed: list = field(default_factory=list)
    cache_clock: pd.Timestamp | None = None
    cache_values: np.ndarray | None = None

    def snapshot(self, at):
        # Consumed ties are still excluded by the original strict-time contract.
        # New rows arrive at/after this clock, so a same-clock cache stays valid.
        if at == self.cache_clock:
            return self.cache_values.copy()
        recent_days, delta_days, gap_days = self.windows
        low = at - pd.Timedelta(days=recent_days)
        prior_low = at - pd.Timedelta(days=delta_days)
        gap_low = at - pd.Timedelta(days=gap_days)
        events = [event for event in self.events if event["at"] < at]
        starts = [when for when in self.starts if when < at]
        completed = [session for session in self.completed if session["available_at"] < at]
        complete = [session for session in completed if session["started_at"] >= low]
        orders = [order for order in self.orders if order["at"] < at]
        recent = [order for order in orders if order["at"] >= low]
        prior = [order for order in orders if prior_low <= order["at"] < low]
        times = [order["at"] for order in orders]
        purchase_gaps = [_days(b, a) for a, b in zip(times, times[1:])]
        visit_gaps = [_days(b, a) for a, b in zip(starts, starts[1:]) if b >= gap_low]
        last = max(completed, key=lambda session: (session["ended_at"], session["started_at"], session["session"])) if completed else None
        local_start = last["started_at"].tz_convert("Asia/Seoul") if last else None
        spend, old_spend = _amount(recent), _amount(prior)
        units = sum(order["units"] for order in recent)
        old_units = sum(order["units"] for order in prior)
        products = {pid for order in recent for pid in order["products"]}
        old_products = {pid for order in prior for pid in order["products"]}
        nonpurchase = [session for session in complete if not session["purchases"]]
        result = dict.fromkeys(EXPANDED_FEATURES, math.nan)
        result.update(
            visit_recency_days=_days(at, starts[-1]) if starts else math.nan,
            purchase_recency_days=_days(at, times[-1]) if times else math.nan,
            observed_tenure_days=_days(at, starts[0]) if starts else math.nan,
            session_count_30d=sum(when >= low for when in starts),
            view_count_30d=sum(event["action"] == "view" and event["at"] >= low for event in events),
            purchase_count_30d=len(recent), net_spend_30d=spend,
            mean_net_order_value_30d=_divide(spend, len(recent)),
            mean_session_duration_minutes_30d=_mean([
                (session["ended_at"] - session["started_at"]).total_seconds() / 60 for session in complete]),
            views_per_session_30d=_divide(sum(session["views"] for session in complete), len(complete)),
            orders_per_session_30d=_divide(sum(session["purchases"] for session in complete), len(complete)),
            mean_purchase_gap_days=_mean(purchase_gaps),
            intervisit_start_gap_mean_days_90d=_mean(visit_gaps),
            intervisit_start_gap_cv_90d=_divide(float(np.std(visit_gaps, ddof=1)), _mean(visit_gaps)) if len(visit_gaps) >= 2 else math.nan,
            unique_purchase_products_30d=len(products), purchased_qty_30d=units,
            mean_basket_qty_30d=_divide(units, len(recent)),
            last_session_min_viewed_price=min(last["view_prices"]) if last and last["view_prices"] else math.nan,
            last_session_max_viewed_price=max(last["view_prices"]) if last and last["view_prices"] else math.nan,
            last_session_start_hour=local_start.hour if local_start is not None else math.nan,
            last_session_start_weekday=local_start.weekday() if local_start is not None else math.nan,
            orders_delta_30d=len(recent) - len(prior), net_spend_delta_30d=spend - old_spend,
            mean_net_order_value_delta_30d=_divide(spend, len(recent)) - _divide(old_spend, len(prior)),
            unique_purchase_products_delta_30d=len(products) - len(old_products),
            purchased_qty_delta_30d=units - old_units,
            cart_adds_per_nonpurchase_session_30d=_divide(sum(session["cart"] for session in nonpurchase), len(nonpurchase)),
            cart_removes_per_nonpurchase_session_30d=_divide(sum(session["remove"] for session in nonpurchase), len(nonpurchase)),
        )
        # Capture does not prove a full rolling window before observation begins.
        for name in EXPANDED_FEATURES:
            if name in _DELTA_FEATURES:
                covered = prior_low >= self.observation_start
            elif name in _GAP_FEATURES:
                covered = gap_low >= self.observation_start
            elif name.endswith("_30d"):
                covered = low >= self.observation_start
            else:
                covered = True
            if not covered or not self.checkout_proxy and name in CHECKOUT_PROXY_FEATURES:
                result[name] = math.nan
        values = np.asarray([result[name] for name in EXPANDED_FEATURES], dtype=float)
        if np.isinf(values).any():
            raise ValueError("source39 cannot contain infinite values")
        self.cache_clock, self.cache_values = at, values
        return values.copy()


def build_source_observations(rows, demonstration_records, *, metadata=None):
    """Return raw39 aligned with every original action/END demonstration.

    Only consumed source rows feed a snapshot, and only timestamps ``< record.at``
    enter its summaries. ``target_at`` is checked for alignment, never used as a
    feature clock. A checkout's payload becomes history after its one prediction.
    """
    required = {"user_id", "session", "at", "action", "product_id", "price_minor", "source_event_id"}
    if not required <= set(rows) or rows.empty:
        raise ValueError("prepared Cosmetics source rows are required")
    if not {"user_id", "session", "at", "target_at", "step", "action"} <= set(demonstration_records):
        raise ValueError("original demonstration record keys are required")
    metadata = dict(metadata or {})
    windows = tuple(feature_window_days(metadata))
    proxy = metadata.get("source_checkout_proxy", True)
    if type(proxy) is not bool:
        raise ValueError("source_checkout_proxy must be boolean")
    observed_actions = set(metadata.get("observed_actions", ACTIONS[:-1]))
    if not observed_actions or not observed_actions <= set(ACTIONS[:-1]):
        raise ValueError("observed_actions must declare supported source shopping actions")
    observed_prices = metadata.get("observed_prices", True)
    if type(observed_prices) is not bool:
        raise ValueError("observed_prices must be boolean")
    source = rows.sort_values(["user_id", "at", "source_event_id"], kind="stable").copy()
    source["at"] = pd.to_datetime(source["at"], utc=True, errors="raise")
    if source["at"].isna().any() or not source.action.isin(ACTIONS[:-1]).all():
        raise ValueError("source timestamps and shopping actions must be known")
    if not source.action.isin(observed_actions).all():
        raise ValueError("source rows contain an action outside their observation support")
    start = _utc(metadata.get("observation_start_at", source["at"].min()))
    end = _utc(metadata.get("observation_end_at", source["at"].max()))
    if end < start or source["at"].lt(start).any() or source["at"].gt(end).any():
        raise ValueError("source rows disagree with the declared observation interval")
    records = iter(demonstration_records.itertuples(index=False))
    vectors, cursor, checkout_groups, collapsed_rows, missing_price_groups = [], 0, 0, 0, 0

    def emit(uid, sid, at, target, step, action, history):
        nonlocal cursor
        try:
            record = next(records)
        except StopIteration as error:
            raise ValueError("source replay has more records than the demonstration frame") from error
        if (str(record.user_id) != str(uid) or str(record.session) != str(sid)
                or _utc(record.at) != at or _utc(record.target_at) != target
                or int(record.step) != step or int(record.action) != action):
            raise ValueError(f"source39 alignment mismatch at demonstration row {cursor}")
        vectors.append(history.snapshot(at))
        cursor += 1

    for uid, user in source.groupby("user_id", sort=True):
        history = _Prefix(start, end, windows, proxy)
        for sid, session in user.groupby("session", sort=False):
            sequence = list(session.itertuples(index=False))
            at, step, index = sequence[0].at, 0, 0
            current = {"session": str(sid), "started_at": at, "views": 0, "view_prices": [],
                       "purchases": 0, "cart": 0, "remove": 0}
            while index < len(sequence):
                event = sequence[index]
                stop = index + 1
                if event.action == "purchase":
                    while stop < len(sequence) and sequence[stop].action == "purchase" and sequence[stop].at == event.at:
                        stop += 1
                emit(uid, sid, at, event.at, step, ACTIONS.index(event.action), history)
                if not step:
                    history.starts.append(event.at)
                prices = [float(item.price_minor) if pd.notna(item.price_minor) and np.isfinite(item.price_minor) and item.price_minor > 0 else None
                          for item in sequence[index:stop]]
                if event.action == "purchase":
                    checkout_groups += 1
                    collapsed_rows += stop - index - 1
                    missing_price_groups += any(price is None for price in prices)
                    history.orders.append({"at": event.at, "units": stop - index,
                        "products": tuple(str(item.product_id) for item in sequence[index:stop]),
                        "amount": sum(prices) if all(price is not None for price in prices) else None})
                    current["purchases"] += 1
                else:
                    current[event.action if event.action != "view" else "views"] += 1
                    if event.action == "view" and prices[0] is not None:
                        current["view_prices"].append(prices[0])
                history.events.append({"at": event.at, "action": event.action})
                at = sequence[stop - 1].at
                step, index = step + 1, stop
            known_at = at + pd.Timedelta(minutes=30)
            emit(uid, sid, at, known_at, step, 4, history)
            if known_at <= end:
                history.completed.append({**current, "ended_at": at, "available_at": known_at})
    if next(records, None) is not None or cursor != len(demonstration_records):
        raise ValueError("demonstration frame has records beyond the source replay")
    unsupported = set(UNOBSERVED_CHANNEL_FEATURES) | set(UNKNOWN_INITIAL_CART_FEATURES)
    if "cart" not in observed_actions:
        unsupported.add("cart_adds_per_nonpurchase_session_30d")
    if "remove" not in observed_actions:
        unsupported.add("cart_removes_per_nonpurchase_session_30d")
    if not observed_prices:
        unsupported.update(("net_spend_30d", "mean_net_order_value_30d", "net_spend_delta_30d",
                            "mean_net_order_value_delta_30d", "last_session_min_viewed_price",
                            "last_session_max_viewed_price"))
    if metadata.get("clock_resolution") == "day":
        unsupported.update(("mean_session_duration_minutes_30d", "last_session_start_hour"))
    if not proxy:
        unsupported.update(CHECKOUT_PROXY_FEATURES)
    raw = np.asarray(vectors, dtype=float).reshape(-1, len(EXPANDED_FEATURES))
    raw[:, [name in unsupported for name in EXPANDED_FEATURES]] = np.nan
    audit = {
        "feature_version": "kit39-source-observed-prefix-v1",
        "feature_names": list(EXPANDED_FEATURES), "features": 39, "records": cursor,
        "source_rows": len(source), "source_users": source.user_id.nunique(),
        "observed_actions": sorted(observed_actions), "observed_prices": observed_prices,
        "clock_resolution": metadata.get("clock_resolution", "source timestamp"),
        "source_measurement": metadata.get("tianchi_measurement"),
        "observation_start_at": start.isoformat(), "observation_end_at": end.isoformat(),
        "time_windows_days": dict(zip(("recent", "delta", "visit_gap"), windows)),
        "alignment_fields": ["user_id", "session", "at", "target_at", "step", "action"],
        "alignment_exact": True, "snapshot_rule": "consumed source prefix only; timestamps strictly less than record.at; no target clock or epsilon",
        "session_completion_rule": "observed END payload, known_at=last source action+30minutes; known_at strictly before snapshot and within capture",
        "unobserved_channel_features": list(UNOBSERVED_CHANNEL_FEATURES),
        "unknown_initial_cart_features": list(UNKNOWN_INITIAL_CART_FEATURES),
        "supported_mask": [name not in unsupported for name in EXPANDED_FEATURES],
        "observed_feature_names": [name for name in EXPANDED_FEATURES
                                   if name not in unsupported and name not in CHECKOUT_PROXY_FEATURES],
        "proxy_feature_names": [name for name in EXPANDED_FEATURES
                                if proxy and name in CHECKOUT_PROXY_FEATURES and name not in unsupported],
        "unsupported_feature_names": [name for name in EXPANDED_FEATURES if name in unsupported],
        "feature_roles": {name: ("unobserved" if name in unsupported else
                                 "checkout_proxy" if name in CHECKOUT_PROXY_FEATURES else "observed")
                          for name in EXPANDED_FEATURES},
        "checkout_proxy_enabled": proxy, "checkout_proxy_features": list(CHECKOUT_PROXY_FEATURES),
        "checkout_proxy_definition": "one contiguous same-user/session/timestamp purchase block is one checkout; each raw purchase row is one observed item token; observed-price sum is not verified net payment or true quantity",
        "checkout_groups": checkout_groups, "collapsed_purchase_rows": collapsed_rows,
        "checkout_groups_with_unknown_price": missing_price_groups,
        "signup": "unobserved; no signup time is fabricated; observed tenure uses first observed past session",
        "price_scope": "consumed event prices only; no catalog lookup or future price fill",
        "reference_timezone": "Asia/Seoul, same convention as KIT39; not an identified consumer local timezone",
        "nonmissing_per_feature": np.isfinite(raw).sum(axis=0).tolist(),
    }
    audit["feature_counts"] = {
        "observed": len(audit["observed_feature_names"]),
        "checkout_proxy": len(audit["proxy_feature_names"]),
        "unobserved": len(audit["unsupported_feature_names"]),
    }
    return raw, audit
