"""Observed behavior summaries; no labels, assignments or simulator diagnostics.

All event histories are strict prefixes (event_at < as_of_at). The two retained
review auxiliary outputs preserve their historical counting and exposure rules.
New product-review outputs require raw5. Legacy other event types can be read
but do not create active features or observation diagnostics.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .time_spec import feature_window_days
from .session_observations import session_availability


BEHAVIOR_FEATURE_VERSION = "behavior_review_cart_v3"
AUXILIARY_FEATURES = (
    "review_count_30d", "mean_review_rating_30d",
)
NEW_EVENT_FEATURES = (
    "cart_adds_per_nonpurchase_session_30d", "cart_removes_per_nonpurchase_session_30d",
    "review_last_order_mean_rating", "review_recency_days", "review_non5_share_30d",
)
BEHAVIOR_FEATURES = AUXILIARY_FEATURES + NEW_EVENT_FEATURES
_DAY = 86_400_000_000_000


def _clock(values):
    result = pd.to_datetime(values, utc=True, errors="raise")
    return result.dt.as_unit("ns") if isinstance(result, pd.Series) else result.as_unit("ns")


def _groups(frame):
    return {uid: group for uid, group in frame.groupby("customer_id", sort=False)}


def _product_reviews(events, purchases, items):
    """Keep observed raw5 purchase/product/line references, once per order/SKU."""
    columns = ["customer_id", "event_at", "related_purchase_event_id", "product_id", "order_line_no", "rating"]
    required = {"rating_kind", *columns}
    if not required <= set(events):
        return pd.DataFrame(columns=columns)
    reviews = events.loc[events.event_type.eq("review") & events.rating_kind.eq("product_star_scenario"), columns].copy()
    if reviews.empty or purchases.empty or items.empty:
        return pd.DataFrame(columns=columns)
    reviews["rating"] = pd.to_numeric(reviews.rating, errors="coerce")
    reviews = reviews.loc[reviews.rating.isin([1, 2, 3, 4, 5]) & reviews.product_id.notna()]
    linked = reviews.merge(purchases[["purchase_event_id", "order_id", "customer_id", "event_at"]],
        left_on=["related_purchase_event_id", "customer_id"], right_on=["purchase_event_id", "customer_id"],
        how="inner", suffixes=("", "_purchase"), validate="many_to_one")
    linked = linked.loc[linked.event_at_purchase.le(linked.event_at)]
    linked = linked.merge(items[["order_id", "product_id", "line_no"]].drop_duplicates(),
        left_on=["order_id", "product_id", "order_line_no"], right_on=["order_id", "product_id", "line_no"],
        how="inner", validate="many_to_one")
    # Repeated quantities/lines of one SKU cannot create additional review weight.
    return (linked.sort_values("event_at", kind="stable")
            .drop_duplicates(["related_purchase_event_id", "product_id"])[columns])


def build_behavior_features(tables, snapshots):
    """Return customer_id/as_of_at, seven features and a product-review mask.

    Session denominators use started_at >= t0-30d and ended_at < t0.

    Review means/fractions first aggregate distinct arrived products per purchase,
    then give each purchase equal weight. Latest review-clock ties average those
    purchases. No-history ages and empty denominators are NaN, observed zeros are
    zero. The current session's future purchase outcome is never read.
    """
    if not {"customer_id", "as_of_at"} <= set(snapshots):
        raise ValueError("Behavior snapshots require customer_id and as_of_at")
    points = snapshots[["customer_id", "as_of_at"]].copy()
    points["as_of_at"] = _clock(points.as_of_at)
    if points.isna().any().any() or points.duplicated().any():
        raise ValueError("Behavior snapshots require known, unique customer/time pairs")
    customers = tables["customers"][["customer_id", "signed_up_at"]].copy()
    customers["signed_up_at"] = _clock(customers.signed_up_at)
    if customers.isna().any().any() or customers.customer_id.duplicated().any():
        raise ValueError("Behavior customers require known unique IDs and signup clocks")
    signed = customers.set_index("customer_id").signed_up_at.to_dict()
    if not points.customer_id.isin(signed).all():
        raise ValueError("Behavior snapshot customer is absent from raw customers")
    ids = set(points.customer_id)
    event_columns = [name for name in ("event_id", "customer_id", "session_id", "event_at", "event_type",
        "related_purchase_event_id", "product_id", "order_line_no", "rating", "rating_kind") if name in tables["events"]]
    events = tables["events"].loc[tables["events"].customer_id.isin(ids), event_columns].copy()
    events["event_at"] = _clock(events.event_at)
    if events.event_at.isna().any():
        raise ValueError("Behavior events require known event clocks")
    sessions = session_availability(tables)
    sessions = sessions.loc[sessions.customer_id.isin(ids)].copy()
    for column in ("started_at", "ended_at"):
        sessions[column] = _clock(sessions[column])
    if sessions.started_at.isna().any():
        raise ValueError("Behavior sessions require a known start clock")
    purchase_events = events.loc[events.event_type.eq("purchase"), ["event_id", "customer_id", "session_id", "event_at"]]
    purchases = tables["orders"][["order_id", "purchase_event_id"]].merge(
        purchase_events, left_on="purchase_event_id", right_on="event_id", how="inner", validate="one_to_one")
    metadata = tables.get("_metadata", {})
    recent_days = feature_window_days(metadata)[0]
    schema = metadata.get("raw_schema_version", 0)
    product_review_supported = isinstance(schema, (int, np.integer)) and schema >= 5
    start = metadata.get("observation_start_at")
    start = _clock(start) if start is not None else None
    if start is not None and pd.isna(start):
        raise ValueError("Behavior observation start must be known when supplied")
    reviews = _product_reviews(events, purchases, tables["order_items"]) if product_review_supported else events.iloc[:0]
    eg = _groups(events.sort_values(["customer_id", "event_at"], kind="stable"))
    sg = _groups(sessions.sort_values(["ended_at", "started_at", "session_id"], kind="stable"))
    og, rg = _groups(purchases), _groups(reviews)
    # Cache sorted customer arrays once; snapshots never scan another customer's history.
    cache = {}
    empty_events, empty_sessions = events.iloc[:0], sessions.iloc[:0]
    empty_orders, empty_reviews = purchases.iloc[:0], reviews.iloc[:0]
    for uid in ids:
        ev = eg.get(uid, empty_events)
        se = sg.get(uid, empty_sessions)
        order = og.get(uid, empty_orders)
        rev = rg.get(uid, empty_reviews)
        cache[uid] = {
            "times": ev.event_at.array.asi8,
            "types": ev.event_type.to_numpy(), "session": ev.session_id.to_numpy(),
            "rating": pd.to_numeric(ev.get("rating", pd.Series(index=ev.index, dtype=float)), errors="coerce").to_numpy(dtype=float, na_value=np.nan),
            "session_starts": se.started_at.array.asi8, "session_ends": se.ended_at.array.asi8,
            "session_available": se.completion_available_at.array.asi8,
            "session_ids": se.session_id.to_numpy(),
            "order_times": order.event_at.array.asi8, "order_sessions": order.session_id.to_numpy(),
            "reviews": rev,
        }
    rows = []
    for point in points.itertuples(index=False):
        uid, t0 = point.customer_id, point.as_of_at
        c, clock, low = cache[uid], t0.value, t0.value - recent_days * _DAY
        past_end = np.searchsorted(c["times"], clock, side="left")
        window_start = np.searchsorted(c["times"], low, side="left")
        window = slice(window_start, past_end)
        kinds = c["types"][window]
        result = dict(zip(BEHAVIOR_FEATURES, [np.nan] * len(BEHAVIOR_FEATURES)))
        # Legacy outputs deliberately count every review row/rating, including raw4.
        result["review_count_30d"] = float(np.count_nonzero(kinds == "review"))
        stars = c["rating"][window][kinds == "review"]
        if np.isfinite(stars).any():
            result["mean_review_rating_30d"] = float(np.nanmean(stars))
        born = t0 > signed[uid]
        covered30 = born and (start is None or start <= max(t0 - pd.Timedelta(days=recent_days), signed[uid]))
        if not covered30:
            result.update(dict.fromkeys(AUXILIARY_FEATURES, np.nan))
        completed = ((c["session_starts"] < clock) & (c["session_available"] < clock)
                     & (c["session_available"] != pd.NaT.value))
        complete30 = completed & (c["session_starts"] >= low)
        session_ids = c["session_ids"][complete30]
        counts = Counter(zip(c["types"][:past_end], c["session"][:past_end]))
        bought_sessions = set(c["order_sessions"][c["order_times"] < clock])
        nonpurchase = [sid for sid in session_ids if sid not in bought_sessions]
        if nonpurchase and covered30:
            result["cart_adds_per_nonpurchase_session_30d"] = sum(counts["cart", sid] for sid in nonpurchase) / len(nonpurchase)
            result["cart_removes_per_nonpurchase_session_30d"] = sum(counts["remove_from_cart", sid] for sid in nonpurchase) / len(nonpurchase)
        if product_review_supported and born:
            rev = c["reviews"]
            past_reviews = rev.loc[rev.event_at.lt(t0)]
            if len(past_reviews):
                by_order = past_reviews.groupby("related_purchase_event_id", sort=False).agg(mean=("rating", "mean"), last=("event_at", "max"))
                last = by_order["last"].max()
                result["review_last_order_mean_rating"] = by_order.loc[by_order["last"].eq(last), "mean"].mean()
                result["review_recency_days"] = (t0 - last).total_seconds() / 86400
                if covered30:
                    recent = past_reviews.loc[past_reviews.event_at.ge(t0 - pd.Timedelta(days=recent_days))]
                    if len(recent):
                        result["review_non5_share_30d"] = recent.assign(non5=recent.rating.lt(5)).groupby("related_purchase_event_id", sort=False).non5.mean().mean()
        rows.append({"customer_id": uid, "as_of_at": t0, **result,
            "behavior_product_review_observed": bool(product_review_supported and born)})
    diagnostics = ["behavior_product_review_observed"]
    out = pd.DataFrame(rows, columns=["customer_id", "as_of_at", *BEHAVIOR_FEATURES, *diagnostics])
    out["as_of_at"] = _clock(out.as_of_at)
    out[list(BEHAVIOR_FEATURES)] = out[list(BEHAVIOR_FEATURES)].astype(float)
    return out
