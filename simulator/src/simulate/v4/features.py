"""Point-in-time KIT39 predictors from canonical raw records.

Labels and experiment assignments remain separate from the registered predictors.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .behavior_features import BEHAVIOR_FEATURES, BEHAVIOR_FEATURE_VERSION, build_behavior_features
from .data.loader import _outcome_days, load_tables

EVENTS = ("view", "cart", "remove_from_cart", "purchase")
KIT_FEATURES = (
    "visit_recency_days", "purchase_recency_days", "observed_tenure_days", "session_count_30d",
    "view_count_30d", "purchase_count_30d", "net_spend_30d", "mean_net_order_value_30d",
    "mean_session_duration_minutes_30d", "views_per_session_30d", "orders_per_session_30d",
    "mean_purchase_gap_days", "intervisit_start_gap_mean_days_90d", "intervisit_start_gap_cv_90d",
    "unique_purchase_products_30d", "purchased_qty_30d", "mean_basket_qty_30d", "cart_qty_current",
    "cart_sku_current", "cart_gross_value_current", "last_session_min_viewed_price",
    "last_session_max_viewed_price", "last_session_start_hour", "last_session_start_weekday",
    "coupon_uses_30d", "coupon_redemption_recency_days", "coupon_discount_amount_30d",
    "orders_delta_30d", "net_spend_delta_30d", "mean_net_order_value_delta_30d",
    "unique_purchase_products_delta_30d", "purchased_qty_delta_30d",
)
EXPANDED_FEATURES = (*KIT_FEATURES, *BEHAVIOR_FEATURES)
EXPANDED_FEATURE_VERSION = "kit39-review-cart-v3"

def _utc(value):
    return pd.to_datetime(value, utc=True, errors="raise")


def _canonical(tables):
    return "customer_id" in tables["customers"] and "signed_up_at" in tables["customers"]


def _prepared(tables):
    result = {name: tables[name].copy() for name in ("customers", "products", "sessions", "events", "orders", "order_items", "coupons")}
    fields = {"customers": ["signed_up_at"], "sessions": ["started_at", "ended_at"],
              "events": ["event_at"], "coupons": ["sent_at", "expires_at"]}
    for name, columns in fields.items():
        for col in columns:
            result[name][col] = _utc(result[name][col])
    purchase = result["events"].loc[result["events"].event_type.eq("purchase"), ["event_id", "customer_id", "session_id", "event_at"]]
    result["orders"] = result["orders"].merge(purchase, left_on="purchase_event_id", right_on="event_id", validate="one_to_one")
    return result


def _groups(frame, key):
    return {key_value: group for key_value, group in frame.groupby(key, sort=False)}


def _divide(a, b):
    return float(a / b) if b else np.nan


def _label_rows(tables, as_of_at=None):
    customers = tables["customers"]
    labels = tables.get("labels")
    if labels is not None:
        aliases = {"user_id": "customer_id", "snapshot_at": "as_of_at"}
        labels = labels.rename(columns={old: new for old, new in aliases.items() if new not in labels})
    if as_of_at is not None:
        cutoff = _utc(as_of_at)
        snapshots = pd.DataFrame({"customer_id": customers.customer_id, "as_of_at": cutoff})
        if labels is not None and {"customer_id", "as_of_at", "snapshot_id"} <= set(labels):
            keys = labels.loc[_utc(labels.as_of_at).eq(cutoff), ["customer_id", "snapshot_id"]]
            snapshots = snapshots.merge(keys, on="customer_id", how="left", validate="one_to_one")
        return snapshots
    if labels is not None:
        if "as_of_at" not in labels:
            raise ValueError("Labels must specify as_of_at; raw history is not an outcome definition")
        labels["as_of_at"] = _utc(labels.as_of_at)
        days = _outcome_days(labels, tables.get("_metadata"))
        labels["outcome_window_days"] = days
        revenue_column = f"future_revenue_{days}d"
        if revenue_column not in labels and "future_revenue" in labels:
            labels[revenue_column] = labels.future_revenue
        if labels.duplicated(["customer_id", "as_of_at"]).any():
            raise ValueError("Duplicate customer/snapshot labels")
        if {"outcome_end_at", "observation_end_at"} <= set(labels):
            immature = _utc(labels.outcome_end_at).gt(_utc(labels.observation_end_at))
            for field in ("churn", "retained", "future_retained", *[name for name in labels if name.startswith("future_revenue")]):
                if field in labels:
                    labels.loc[immature, field] = np.nan
        if {"clv_outcome_end_at", "observation_end_at", "future_revenue_12m"} <= set(labels):
            labels.loc[_utc(labels.clv_outcome_end_at).gt(_utc(labels.observation_end_at)), "future_revenue_12m"] = np.nan
        return labels
    raise ValueError("Supply as_of_at or explicit labels sidecar")


def build_features(tables: dict[str, Any], as_of_at=None) -> pd.DataFrame:
    """Compute KIT39 from strict past records; an explicit cutoff imports only matching snapshot IDs."""
    if as_of_at is not None and (pd.isna(pd.Timestamp(as_of_at)) or pd.Timestamp(as_of_at).tzinfo is None):
        raise ValueError("Feature as_of_at must be a timezone-aware timestamp")
    if not _canonical(tables):
        raise ValueError("Legacy Parquet has no canonical price/cart/coupon contract; regenerate canonical data before feature/training")
    from simulate.v4.simulator.raw_store import cart_at
    labels = _label_rows(tables, as_of_at)
    metadata = tables.get("_metadata", {})
    if labels.as_of_at.isna().any():
        raise ValueError("Every feature snapshot requires a known as_of_at")
    for key, outside in (("observation_start_at", "before"), ("observation_end_at", "after")):
        if key in metadata:
            boundary = _utc(metadata[key])
            invalid = labels.as_of_at.lt(boundary) if outside == "before" else labels.as_of_at.gt(boundary)
            if invalid.any():
                raise ValueError(f"Feature as_of_at is {outside} the dataset observation range")
            if key == "observation_end_at":
                for target in ("churn", "retained", "future_retained", *[name for name in labels if name.startswith("future_revenue")]):
                    if target in labels:
                        labels.loc[labels.as_of_at.eq(boundary), target] = np.nan
    scoped = tables
    if labels.customer_id.nunique() < len(tables["customers"]):
        ids = set(labels.customer_id)
        scoped = dict(tables)
        for name in ("customers", "sessions", "events", "coupons"):
            scoped[name] = tables[name].loc[tables[name].customer_id.isin(ids)]
        scoped["orders"] = tables["orders"].loc[tables["orders"].purchase_event_id.isin(scoped["events"].event_id)]
        scoped["order_items"] = tables["order_items"].loc[tables["order_items"].order_id.isin(scoped["orders"].order_id)]
    raw = _prepared(scoped)
    behavior = build_behavior_features(scoped, labels[["customer_id", "as_of_at"]])
    behavior_by_snapshot = behavior.set_index(["customer_id", "as_of_at"]).to_dict("index")
    products = raw["products"].set_index("product_id").unit_price_minor.to_dict()
    customers = raw["customers"].set_index("customer_id")
    observation_start = _utc(metadata.get("observation_start_at"))
    rows = []
    for t0, snapshots in labels.groupby("as_of_at", sort=True):
        snapshot_ids = snapshots.set_index("customer_id")["snapshot_id"].dropna().to_dict() if "snapshot_id" in snapshots else {}
        t30, t60, t90 = (t0 - pd.Timedelta(days=n) for n in (30, 60, 90))
        ev = raw["events"].loc[raw["events"].event_at.lt(t0) & raw["events"].event_type.isin(EVENTS)].sort_values(["event_at", "event_id"])
        ss = raw["sessions"].loc[raw["sessions"].started_at.lt(t0)]
        od = raw["orders"].loc[raw["orders"].event_at.lt(t0)]
        cp = raw["coupons"].loc[raw["coupons"].sent_at.lt(t0)]
        eg, sg, og, cg = (_groups(frame, "customer_id") for frame in (ev, ss, od, cp))
        ig = _groups(raw["order_items"], "order_id")
        for uid in snapshots.customer_id:
            uev, uss, uod, ucp = (groups.get(uid, empty.iloc[:0]) for groups, empty in ((eg, ev), (sg, ss), (og, od), (cg, cp)))
            r = {"customer_id": uid, "user_id": uid, "as_of_at": t0,
                 "snapshot_id": snapshot_ids.get(uid, f"{uid}|{t0.isoformat()}"), "reference_week": (t0.tz_convert("Asia/Seoul") - pd.Timedelta(days=t0.tz_convert("Asia/Seoul").weekday())).date().isoformat()}
            o30, prior = uod.loc[uod.event_at.ge(t30)], uod.loc[uod.event_at.ge(t60) & uod.event_at.lt(t30)]
            item30 = pd.concat([ig[k] for k in o30.order_id], ignore_index=True) if len(o30) else raw["order_items"].iloc[:0]
            itemprior = pd.concat([ig[k] for k in prior.order_id], ignore_index=True) if len(prior) else raw["order_items"].iloc[:0]
            completed = uss.loc[uss.ended_at.lt(t0)]
            complete = completed.loc[completed.started_at.ge(t30)].sort_values("ended_at")
            r["visit_recency_days"] = (t0 - uss.started_at.max()).total_seconds() / 86400 if len(uss) else np.nan
            r["purchase_recency_days"] = (t0 - uod.event_at.max()).total_seconds() / 86400 if len(uod) else np.nan
            r["observed_tenure_days"] = (t0 - uss.started_at.min()).total_seconds() / 86400 if len(uss) else np.nan
            r["session_count_30d"] = int(uss.started_at.ge(t30).sum())
            r["view_count_30d"] = int((uev.event_at.ge(t30) & uev.event_type.eq("view")).sum())
            r["purchase_count_30d"] = len(o30)
            r["net_spend_30d"] = int(o30.paid_minor.sum())
            r["mean_net_order_value_30d"] = o30.paid_minor.mean()
            r["mean_session_duration_minutes_30d"] = ((complete.ended_at - complete.started_at).dt.total_seconds() / 60).mean()
            in_complete = uev.session_id.isin(complete.session_id)
            r["views_per_session_30d"] = _divide((in_complete & uev.event_type.eq("view")).sum(), len(complete))
            r["orders_per_session_30d"] = _divide(uod.session_id.isin(complete.session_id).sum(), len(complete))
            gaps = uod.sort_values(["event_at", "event_id"]).event_at.diff().dt.total_seconds() / 86400
            r["mean_purchase_gap_days"] = gaps.mean()
            starts = uss.sort_values("started_at").started_at
            visit_gaps = starts.diff().dt.total_seconds().loc[starts.ge(t90)] / 86400
            r["intervisit_start_gap_mean_days_90d"] = visit_gaps.mean()
            r["intervisit_start_gap_cv_90d"] = _divide(visit_gaps.std(ddof=1), visit_gaps.mean()) if visit_gaps.notna().sum() >= 2 else np.nan
            r["unique_purchase_products_30d"] = item30.product_id.nunique()
            r["purchased_qty_30d"] = int(item30.quantity.sum())
            r["mean_basket_qty_30d"] = _divide(r["purchased_qty_30d"], len(o30))
            subset = {**tables, "customers": customers.loc[[uid]].reset_index(), "events": uev, "orders": uod,
                      "order_items": pd.concat([ig[k] for k in uod.order_id], ignore_index=True) if len(uod) else raw["order_items"].iloc[:0]}
            cart = cart_at(subset, uid, t0)
            r["cart_qty_current"] = sum(cart.values())
            r["cart_sku_current"] = sum(qty > 0 for qty in cart.values())
            r["cart_gross_value_current"] = sum(qty * products[pid] for pid, qty in cart.items())
            last = completed.sort_values(["ended_at", "started_at", "session_id"]).iloc[-1] if len(completed) else None
            last_views = uev.loc[uev.session_id.eq(last.session_id) & uev.event_type.eq("view")] if last is not None else uev.iloc[:0]
            r["last_session_min_viewed_price"] = last_views.unit_price_minor.min()
            r["last_session_max_viewed_price"] = last_views.unit_price_minor.max()
            local_start = last.started_at.tz_convert("Asia/Seoul") if last is not None else None
            r["last_session_start_hour"] = local_start.hour if local_start is not None else np.nan
            r["last_session_start_weekday"] = local_start.weekday() if local_start is not None else np.nan
            redeemed = uod.loc[uod.coupon_id.notna()]
            redeemed30 = redeemed.loc[redeemed.event_at.ge(t30)]
            r["coupon_uses_30d"] = redeemed30.coupon_id.nunique()
            r["coupon_redemption_recency_days"] = (t0 - redeemed.event_at.max()).total_seconds() / 86400 if len(redeemed) else np.nan
            r["coupon_discount_amount_30d"] = int(redeemed30.discount_minor.sum())
            r["orders_delta_30d"] = len(o30) - len(prior)
            r["net_spend_delta_30d"] = int(o30.paid_minor.sum() - prior.paid_minor.sum())
            r["mean_net_order_value_delta_30d"] = o30.paid_minor.mean() - prior.paid_minor.mean()
            r["unique_purchase_products_delta_30d"] = item30.product_id.nunique() - itemprior.product_id.nunique()
            r["purchased_qty_delta_30d"] = int(item30.quantity.sum() - itemprior.quantity.sum())
            # PRD compatibility outputs are separate from the fixed KIT32 predictor set.
            r.update(behavior_by_snapshot[(uid, t0)])
            prior_sessions = (uss.started_at.ge(t60) & uss.started_at.lt(t30)).sum()
            for name, recent, previous in (("visit", r["session_count_30d"], prior_sessions), ("orders", len(o30), len(prior)),
                    ("spend", o30.paid_minor.sum(), prior.paid_minor.sum()), ("basket_qty", item30.quantity.sum(), itemprior.quantity.sum()),
                    ("coupon", len(redeemed30), (redeemed.event_at.ge(t60) & redeemed.event_at.lt(t30)).sum())):
                r[f"{name}_change_ratio_30d"] = _divide(recent, previous)
            r["purchase_cycle_anomaly"] = _divide(r["purchase_recency_days"], r["mean_purchase_gap_days"])
            r["journey_stage"] = "no_purchase" if not len(uod) else "first_purchase" if len(uod) == 1 else "repeat_purchase"
            r["event_type_sequence"] = [EVENTS.index(v) + 1 for v in uev.event_type.iloc[-40:]]
            r["inter_event_gap_sequence"] = (uev.event_at.diff().dt.total_seconds().fillna(0).iloc[-40:]).tolist()
            prices = uev.unit_price_minor
            if "offer_unit_price_minor" in uev:
                prices = uev.offer_unit_price_minor.combine_first(prices)
            r["event_price_sequence"] = prices.fillna(0).iloc[-40:].astype(float).tolist()
            inventory, qty_context, value_context = {}, [], []
            purchase_orders = uod.set_index("purchase_event_id").to_dict("index")
            for event in uev.itertuples():
                qty_context.append(sum(inventory.values()))
                value_context.append(sum(q * products[p] for p, q in inventory.items()))
                if event.event_type in ("cart", "remove_from_cart"):
                    inventory[event.product_id] = inventory.get(event.product_id, 0) + int(event.quantity) * (1 if event.event_type == "cart" else -1)
                elif event.event_type == "purchase":
                    order = purchase_orders[event.event_id]
                    if order["checkout_source"] in {"cart", "mixed"}:
                        for item in ig[order["order_id"]].itertuples():
                            held = inventory.get(item.product_id, 0)
                            inventory[item.product_id] = held - min(held, int(item.quantity))
            if {k: v for k, v in inventory.items() if v} != {k: v for k, v in cart.items() if v}:
                raise ValueError("Sequence cart context disagrees with shared cart replay")
            r["event_cart_qty_sequence"] = qty_context[-40:]
            r["event_cart_value_sequence"] = value_context[-40:]
            metadata = tables.get("_metadata", {})
            r["currency"] = metadata.get("currency", "UNSPECIFIED")
            r["amount_unit"] = str(metadata.get("amount_unit", "UNSPECIFIED"))
            r["dataset_id"] = metadata.get("dataset_id", "UNSPECIFIED")
            r["raw_database_sha256"] = metadata.get("raw_database_sha256", "UNSPECIFIED")
            r["source_model_sha256"] = metadata.get("model_sha256", "UNSPECIFIED")
            r["feature_set_version"] = "kit32-v1"
            r["behavior_feature_set_version"] = BEHAVIOR_FEATURE_VERSION
            r["expanded_feature_set_version"] = EXPANDED_FEATURE_VERSION
            r["intervention_policy_id"] = metadata.get("generation_policy_id", metadata.get("policy_id", "one_coupon_valid30d_no_further_send"))

            signup = customers.loc[uid].signed_up_at
            if signup >= t0:
                r.update(dict.fromkeys(EXPANDED_FEATURES, np.nan))
            elif observation_start is not None and signup < observation_start:
                # A capture that starts after signup cannot establish the initial cart.
                for field in KIT_FEATURES:
                    if (field.startswith("cart_")
                            or (field.endswith("_30d") and observation_start > (t60 if "_delta_" in field else t30))
                            or (field.endswith("_90d") and observation_start > t90)):
                        r[field] = np.nan
            rows.append(r)
    out = pd.DataFrame(rows).merge(labels.drop(columns=["user_id", "snapshot_id", "snapshot_at"], errors="ignore"), on=["customer_id", "as_of_at"], how="left", validate="one_to_one")
    numeric = [*EXPANDED_FEATURES, "purchase_cycle_anomaly", *[f"{name}_change_ratio_30d" for name in ("visit", "orders", "spend", "basket_qty", "coupon")]]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="raise").astype("float64").replace([np.inf, -np.inf], np.nan)
    treatment = tables.get("treatments")
    if treatment is not None and len(treatment):
        treatment = treatment.copy()
        if "customer_id" not in treatment:
            treatment["customer_id"] = treatment.user_id
        if "as_of_at" not in treatment:
            treatment["as_of_at"] = treatment.assigned_at
        treatment["as_of_at"] = _utc(treatment.as_of_at)
        keep = [c for c in ("customer_id", "as_of_at", "treatment", "assignment_probability", "experiment_id", "actual_sent", "send_cost_minor") if c in treatment]
        out = out.merge(treatment[keep], on=["customer_id", "as_of_at"], how="left", validate="one_to_one", suffixes=("", "_assignment"))
    return out
