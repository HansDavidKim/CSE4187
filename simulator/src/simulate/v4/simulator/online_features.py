"""Strict-past KIT39 snapshots for a single canonical online customer."""
from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd

from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.session_observations import completion_exclusions
from simulate.v4.time_spec import feature_window_days

DAY = 86_400_000_000_000
SHOPPING = {"view", "cart", "remove_from_cart", "purchase"}


def _time(value):
    if value is None or value is pd.NaT:
        return None
    stamp = value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)
    return None if pd.isna(stamp) else stamp.value


def _number(value):
    return float(value) if value is not None and pd.notna(value) else math.nan


def _mean(values):
    return float(np.mean(values)) if values else math.nan


def _divide(numerator, denominator):
    return numerator / denominator if denominator else math.nan


def _age(delta):
    # Preserve scalar Timedelta.total_seconds() precision in the original feature contract.
    return (delta // 1000) / 86_400_000_000


def observed_snapshot(tables, as_of_at, customer_id=None, product_prices=None, *, left_censored=False):
    """Return the existing 39 predictors without DataFrame work per transition.

    ``tables`` accepts canonical record lists or DataFrames. It must contain the
    complete observed history for this customer, including its initial cart.
    ``left_censored`` retains source-capture window coverage when entry is not signup.
    """
    at = pd.Timestamp(as_of_at)
    if pd.isna(at) or at.tzinfo is None:
        raise ValueError("online feature cutoff must be timezone-aware")
    clock = at.value
    metadata = tables.get("_metadata", {})

    def records(name):
        value = tables[name]
        return value.to_dict("records") if isinstance(value, pd.DataFrame) else value

    customers = records("customers")
    if customer_id is None:
        if len(customers) != 1:
            raise ValueError("online snapshots need one customer or an explicit customer_id")
        customer_id = customers[0]["customer_id"]
    customer = next(row for row in customers if row["customer_id"] == customer_id)
    signed = _time(customer["signed_up_at"])
    if signed >= clock:
        return np.full(len(EXPANDED_FEATURES), np.nan)
    start = _time(metadata.get("observation_start_at"))
    recent_days, delta_days, gap_days = feature_window_days(metadata)
    t30, t60, t90 = (clock - days * DAY for days in (recent_days, delta_days, gap_days))
    all_events = [(row, _time(row["event_at"])) for row in records("events")
                  if row["customer_id"] == customer_id]
    all_events = [(row, when) for row, when in all_events if when < clock]
    events = sorted(((row, when) for row, when in all_events if row["event_type"] in SHOPPING),
                    key=lambda pair: (pair[1], pair[0]["event_id"]))
    sessions = [(row, _time(row["started_at"]), _time(row.get("ended_at")))
                for row in records("sessions") if row["customer_id"] == customer_id
                and _time(row["started_at"]) < clock]
    starts = sorted(began for _, began, _ in sessions)
    excluded = completion_exclusions(metadata)
    diagnostics = tables.get("session_diagnostics") if excluded is None else None
    by_session = None
    if diagnostics is not None:
        diagnostics = diagnostics.to_dict("records") if isinstance(diagnostics, pd.DataFrame) else diagnostics
        by_session = {}
        for row in diagnostics:
            key = (row.get("customer_id"), row.get("session_id"))
            reason, truncated = row.get("termination_reason"), row.get("truncated_session")
            if (any(value is None or pd.isna(value) for value in key) or key in by_session
                    or not isinstance(reason, str) or not reason
                    or not isinstance(truncated, (bool, np.bool_))):
                raise ValueError("session diagnostics require unique known keys, reasons and boolean truncation flags")
            by_session[key] = row
    completed = []
    for row, began, ended in sessions:
        if excluded is not None:
            available = ended + 30 * 60 * 1_000_000_000 if ended is not None and row["session_id"] not in excluded else None
        elif by_session is None:
            available = _time(row.get("completion_available_at", row.get("ended_at")))
        else:
            diagnostic = by_session.get((customer_id, row["session_id"]))
            if diagnostic is None:
                raise ValueError("session diagnostics must cover every observed visit in the scope")
            natural = diagnostic["termination_reason"] in {"causal_policy_end", "donor_length"} and not diagnostic["truncated_session"]
            available = ended + 30 * 60 * 1_000_000_000 if natural and ended is not None else None
        if ended is not None and ended < clock and available is not None and available < clock:
            completed.append((row, began, ended))
    complete30 = [item for item in completed if item[1] >= t30]
    session_ids = {row["session_id"] for row, _, _ in complete30}
    purchases = {row["event_id"]: (row, when) for row, when in events if row["event_type"] == "purchase"}
    orders = [(row, *purchases[row["purchase_event_id"]]) for row in records("orders")
              if row["purchase_event_id"] in purchases]
    orders.sort(key=lambda item: (item[2], item[1]["event_id"]))
    recent = [item for item in orders if item[2] >= t30]
    prior = [item for item in orders if t60 <= item[2] < t30]
    by_order = defaultdict(list)
    for item in records("order_items"):
        by_order[item["order_id"]].append(item)
    recent_items = [item for order, _, _ in recent for item in by_order[order["order_id"]]]
    prior_items = [item for order, _, _ in prior for item in by_order[order["order_id"]]]
    prices = product_prices if product_prices is not None else {
        row["product_id"]: row["unit_price_minor"] for row in records("products")}
    order_by_event = {row["purchase_event_id"]: row for row, _, _ in orders}
    cart = Counter()
    for event, _ in events:
        kind, pid = event["event_type"], event.get("product_id")
        if kind == "cart":
            cart[pid] += int(event["quantity"])
        elif kind == "remove_from_cart":
            cart[pid] -= int(event["quantity"])
        elif kind == "purchase":
            order = order_by_event[event["event_id"]]
            if order["checkout_source"] in {"cart", "mixed"}:
                for item in by_order[order["order_id"]]:
                    cart[item["product_id"]] -= min(cart[item["product_id"]], int(item["quantity"]))
    if any(quantity < 0 for quantity in cart.values()):
        raise ValueError("online cart history contains a negative quantity")
    last = max(completed, key=lambda item: (item[2], item[1], item[0]["session_id"])) if completed else None
    viewed = [_number(row.get("unit_price_minor")) for row, _ in events
              if last is not None and row["session_id"] == last[0]["session_id"] and row["event_type"] == "view"]
    viewed = [value for value in viewed if math.isfinite(value)]
    local_start = pd.Timestamp(last[1], tz="UTC").tz_convert("Asia/Seoul") if last else None
    redeemed = [item for item in orders if pd.notna(item[0].get("coupon_id"))]
    redeemed30 = [item for item in redeemed if item[2] >= t30]
    purchase_times = [when for _, _, when in orders]
    purchase_gaps = [(b-a)/DAY for a, b in zip(purchase_times, purchase_times[1:])]
    visit_gaps = [(b-a)/DAY for a, b in zip(starts, starts[1:]) if b >= t90]
    spend = sum(order["paid_minor"] for order, _, _ in recent)
    old_spend = sum(order["paid_minor"] for order, _, _ in prior)
    units = sum(item["quantity"] for item in recent_items)
    old_units = sum(item["quantity"] for item in prior_items)
    unique = len({item["product_id"] for item in recent_items})
    old_unique = len({item["product_id"] for item in prior_items})
    r = dict(zip(EXPANDED_FEATURES, [math.nan] * len(EXPANDED_FEATURES)))
    r.update(visit_recency_days=_age(clock-starts[-1]) if starts else math.nan,
             purchase_recency_days=_age(clock-purchase_times[-1]) if purchase_times else math.nan,
             observed_tenure_days=_age(clock-starts[0]) if starts else math.nan,
             session_count_30d=sum(began >= t30 for began in starts),
             view_count_30d=sum(when >= t30 and row["event_type"] == "view" for row, when in events),
             purchase_count_30d=len(recent), net_spend_30d=spend,
             mean_net_order_value_30d=_divide(spend, len(recent)),
             mean_session_duration_minutes_30d=_mean([(ended-began)/60_000_000_000 for _, began, ended in complete30]),
             views_per_session_30d=_divide(sum(row["event_type"] == "view" and row["session_id"] in session_ids for row, _ in events), len(complete30)),
             orders_per_session_30d=_divide(sum(event["session_id"] in session_ids for _, event, _ in orders), len(complete30)),
             mean_purchase_gap_days=_mean(purchase_gaps),
             intervisit_start_gap_mean_days_90d=_mean(visit_gaps),
             intervisit_start_gap_cv_90d=_divide(float(np.std(visit_gaps, ddof=1)), _mean(visit_gaps)) if len(visit_gaps) >= 2 else math.nan,
             unique_purchase_products_30d=unique, purchased_qty_30d=units,
             mean_basket_qty_30d=_divide(units, len(recent)), cart_qty_current=sum(cart.values()),
             cart_sku_current=sum(quantity > 0 for quantity in cart.values()),
             cart_gross_value_current=sum(quantity*prices[pid] for pid, quantity in cart.items()),
             last_session_min_viewed_price=min(viewed) if viewed else math.nan,
             last_session_max_viewed_price=max(viewed) if viewed else math.nan,
             last_session_start_hour=local_start.hour if local_start is not None else math.nan,
             last_session_start_weekday=local_start.weekday() if local_start is not None else math.nan,
             coupon_uses_30d=len({order["coupon_id"] for order, _, _ in redeemed30}),
             coupon_redemption_recency_days=_age(clock-max(when for _, _, when in redeemed)) if redeemed else math.nan,
             coupon_discount_amount_30d=sum(order["discount_minor"] for order, _, _ in redeemed30),
             orders_delta_30d=len(recent)-len(prior), net_spend_delta_30d=spend-old_spend,
             mean_net_order_value_delta_30d=_divide(spend,len(recent))-_divide(old_spend,len(prior)),
             unique_purchase_products_delta_30d=unique-old_unique, purchased_qty_delta_30d=units-old_units)
    covered30 = start is None or start <= (t30 if left_censored else max(t30, signed))
    reviews = [(row, when) for row, when in all_events if row["event_type"] == "review"]
    reviews30 = [(row, when) for row, when in reviews if when >= t30]
    ratings = [_number(row.get("rating")) for row, _ in reviews30]
    ratings = [value for value in ratings if math.isfinite(value)]
    if covered30:
        r.update(review_count_30d=float(len(reviews30)), mean_review_rating_30d=_mean(ratings))
        bought_sessions = {event["session_id"] for _, event, _ in orders}
        nonpurchase = session_ids - bought_sessions
        if nonpurchase:
            r["cart_adds_per_nonpurchase_session_30d"] = sum(row["event_type"] == "cart" and row["session_id"] in nonpurchase for row, _ in events)/len(nonpurchase)
            r["cart_removes_per_nonpurchase_session_30d"] = sum(row["event_type"] == "remove_from_cart" and row["session_id"] in nonpurchase for row, _ in events)/len(nonpurchase)
    if metadata.get("raw_schema_version", 0) >= 5:
        valid, seen = [], set()
        for review, when in sorted(reviews, key=lambda pair: pair[1]):
            key = (review.get("related_purchase_event_id"), review.get("product_id"))
            order = order_by_event.get(key[0])
            rating = _number(review.get("rating"))
            if (key in seen or review.get("rating_kind") != "product_star_scenario" or order is None
                    or rating not in (1,2,3,4,5) or purchases[key[0]][1] > when
                    or not any(item["product_id"] == key[1] and item["line_no"] == review.get("order_line_no") for item in by_order[order["order_id"]])):
                continue
            seen.add(key); valid.append((key[0], rating, when))
        grouped = defaultdict(list)
        for key, rating, when in valid:
            grouped[key].append((rating, when))
        if grouped:
            last_clock = max(when for _, _, when in valid)
            r["review_last_order_mean_rating"] = _mean([_mean([rating for rating, _ in values]) for values in grouped.values() if max(when for _, when in values) == last_clock])
            r["review_recency_days"] = _age(clock-last_clock)
            recent_reviews = defaultdict(list)
            for key, rating, when in valid:
                if when >= t30:
                    recent_reviews[key].append(rating < 5)
            if covered30:
                r["review_non5_share_30d"] = _mean([_mean(values) for values in recent_reviews.values()])
    if start is not None and (signed < start or left_censored):
        for name in EXPANDED_FEATURES[:32]:
            if (name.startswith("cart_") or name.endswith("_30d") and start > (t60 if "_delta_" in name else t30)
                    or name.endswith("_90d") and start > t90):
                r[name] = math.nan
    return np.asarray([r[name] for name in EXPANDED_FEATURES], dtype=float)
