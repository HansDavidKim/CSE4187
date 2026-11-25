"""Executable-contract documentation and training-only feature quality summaries."""
from pathlib import Path
import re

import numpy as np
import pandas as pd


# Formulas follow the simulator's registered KIT39 definitions.
DEFINITIONS = {
    "visit_recency_days": ("Days since latest session start", "all past", "no past session", "Visit recency"),
    "purchase_recency_days": ("Days since latest order purchase event", "all past", "no order", "Purchase recency"),
    "observed_tenure_days": ("Days since first observed session start", "all past", "no session", "Observed history length"),
    "session_count_30d": ("Number of session starts", "recent 30d", "incomplete window", "Visit frequency"),
    "view_count_30d": ("Number of view events", "recent 30d", "incomplete window", "Product interest"),
    "purchase_count_30d": ("Number of orders", "recent 30d", "incomplete window", "Purchase frequency"),
    "net_spend_30d": ("Sum of order paid_minor", "recent 30d", "incomplete window", "Paid monetary value"),
    "mean_net_order_value_30d": ("Mean order paid_minor", "recent 30d", "no orders or incomplete window", "Order value"),
    "mean_session_duration_minutes_30d": ("Mean (ended_at-started_at)/60 seconds for completed sessions", "recent 30d starts", "no completed sessions or incomplete window", "Session engagement"),
    "views_per_session_30d": ("Views in completed sessions / completed session count", "recent 30d starts", "zero denominator or incomplete window", "Browsing intensity"),
    "orders_per_session_30d": ("Orders in completed sessions / completed session count", "recent 30d starts", "zero denominator or incomplete window", "Session conversion"),
    "mean_purchase_gap_days": ("Mean interval between chronologically sorted order events", "all past", "fewer than two orders", "Usual purchase cycle"),
    "intervisit_start_gap_mean_days_90d": ("Mean start-to-start gap whose latter session starts in recent 90d", "90d plus preceding start", "insufficient history or gaps", "Visit regularity"),
    "intervisit_start_gap_cv_90d": ("Sample standard deviation of visit gaps / mean gap", "90d plus preceding start", "fewer than two gaps, nonpositive mean or incomplete window", "Visit variability"),
    "unique_purchase_products_30d": ("Distinct product_id in order items", "recent 30d orders", "incomplete window", "Product breadth"),
    "purchased_qty_30d": ("Sum of order-item quantity", "recent 30d orders", "incomplete window", "Purchase volume"),
    "mean_basket_qty_30d": ("Purchased item quantity / order count", "recent 30d", "zero orders or incomplete window", "Basket size"),
    "cart_qty_current": ("Quantity remaining after replaying past cart and checkout events", "signup to cutoff", "incomplete cart history", "Unfinished purchase intent"),
    "cart_sku_current": ("Distinct products with positive cart quantity", "signup to cutoff", "incomplete cart history", "Unfinished product breadth"),
    "cart_gross_value_current": ("Sum of remaining quantity times canonical product unit price", "signup to cutoff", "incomplete cart history", "Unfinished basket value"),
    "last_session_min_viewed_price": ("Minimum viewed unit price in last completed session", "all past completed sessions", "no completed session or no views", "Last-session price context"),
    "last_session_max_viewed_price": ("Maximum viewed unit price in last completed session", "all past completed sessions", "no completed session or no views", "Last-session price context"),
    "last_session_start_hour": ("Asia/Seoul start hour of last completed session", "all past completed sessions", "no completed session", "Activity time of day"),
    "last_session_start_weekday": ("Asia/Seoul start weekday, Monday=0", "all past completed sessions", "no completed session", "Weekly activity context"),
    "coupon_uses_30d": ("Distinct coupon_id on past orders", "recent 30d", "incomplete window", "Observed coupon response"),
    "coupon_redemption_recency_days": ("Days since latest past order using a coupon", "all past", "no past coupon use", "Coupon response recency"),
    "coupon_discount_amount_30d": ("Sum of actual redeemed order discount_minor", "recent 30d", "incomplete window", "Observed coupon savings"),
    "review_count_30d": ("Number of arrived review event rows", "recent 30d", "incomplete window", "Review activity"),
    "mean_review_rating_30d": ("Mean finite rating of arrived review rows", "recent 30d", "no finite ratings or incomplete window", "Observed review ratings"),
    "cart_adds_per_nonpurchase_session_30d": ("Cart-add events in completed nonpurchase sessions / such session count", "recent 30d starts", "zero denominator or incomplete window", "Unconverted cart engagement"),
    "cart_removes_per_nonpurchase_session_30d": ("Cart-removal events in completed nonpurchase sessions / such session count", "recent 30d starts", "zero denominator or incomplete window", "Unconverted cart reconsideration"),
    "review_last_order_mean_rating": ("Mean product rating per order, then mean across orders tied at latest review arrival", "all past arrived valid product reviews", "unsupported schema or no linked reviews", "Latest observed purchase feedback"),
    "review_recency_days": ("Days since latest valid product review arrival", "all past arrived valid product reviews", "unsupported schema or no linked reviews", "Feedback recency"),
    "review_non5_share_30d": ("Share of product ratings below 5 within each order, averaged across orders", "recent 30d arrived valid product reviews", "unsupported schema, no reviews or incomplete window", "Observed below-five feedback"),
    "purchase_cycle_anomaly": ("purchase_recency_days / mean_purchase_gap_days", "all past orders", "unknown or nonpositive average gap", "Purchase-cycle deviation"),
    "journey_stage": ("Past order count capped at 3, meaning 0/1/2/3+ orders", "signup to cutoff", "incomplete lifetime history", "Purchase-stage proxy"),
    "stage_tenure_days": ("Days since signup or the first/second/third order entering the current stage", "signup to cutoff", "incomplete lifetime history", "Time in purchase-stage proxy"),
    "action_group_transitions": ("Count adjacent timestamp groups with different sets of four shopping actions", "recent 30d", "incomplete window", "Action-sequence changes"),
    "last_action_group_run": ("Length of trailing timestamp groups with the same action set", "recent 30d", "incomplete window", "Action-sequence persistence"),
}

CHANGES = {
    "orders": ("purchase_count", "Order frequency"),
    "net_spend": ("net_spend", "Paid monetary value"),
    "mean_net_order_value": ("mean_net_order_value", "Order value"),
    "unique_purchase_products": ("unique_purchase_products", "Product breadth"),
    "purchased_qty": ("purchased_qty", "Purchase volume"),
}
DEFINITIONS.update({
    "history_session_count": ("Count of session starts strictly before cutoff", "observed past", "unknown history", "Long-history visit frequency"),
    "history_active_days": ("Distinct UTC dates of observed session starts", "observed past", "unknown history", "Long-history visit regularity"),
    "history_session_rate_30d": ("Past session count times 30 / days since max(signup, observation start)", "observed past", "nonpositive observed age", "Exposure-adjusted visit frequency"),
    "session_count_90d": ("Session starts in [cutoff-90d, cutoff)", "recent 90d", "incomplete window", "Medium-term visit frequency"),
    "history_order_count": ("Count of actual orders joined to past purchase events", "observed past", "unknown history", "Long-history purchase frequency"),
})
for name in ("view", "cart", "remove", "purchase"):
    DEFINITIONS[f"history_{name}_share"] = (
        f"Observed {name} shopping actions / all four observed shopping actions", "observed past",
        "no shopping actions", "Long-history action composition")
for name, (base, meaning) in CHANGES.items():
    delta = "orders_delta_30d" if name == "orders" else f"{base}_delta_30d"
    DEFINITIONS[delta] = (f"Recent {base}_30d minus previous 30d value", "two adjacent 30d windows",
                          "incomplete window or undefined constituent mean", f"Change in {meaning.lower()}")
    DEFINITIONS[f"{name}_relative_change_30d"] = (
        "(Recent 30d minus previous 30d) / previous 30d", "two adjacent 30d windows",
        "unknown or nonpositive previous value", f"Relative change in {meaning.lower()}")


def write_dictionary(columns, out, *, time_windows_days=None):
    """Write a complete formula/window/missingness/business mapping."""
    windows = time_windows_days or {}
    recent, gap = windows.get("recent_days", 30), windows.get("gap_days", 90)
    lines = ["# Retention feature dictionary", "", "All inputs are strictly before the timezone-aware cutoff.",
             "Known empty count/sum windows are zero. Unknown windows and undefined denominators are null.",
             "Money remains in source minor units. Time features use days or the units named in each formula.",
             "Train-only quantile clipping precedes median imputation and missing indicators.", "",
             "| Feature | Formula | Window | Null condition | Meaning |",
             "| --- | --- | --- | --- | --- |"]
    if (recent, gap) != (30, 90):
        lines[2:2] = [f"Weekly inputs use {recent}-day recent/change windows and {gap}-day gap windows. "
                       "The registered _30d/_90d column names remain compatibility aliases."]
    for name in columns:
        if name not in DEFINITIONS:
            raise ValueError(f"Missing feature definition: {name}")
        values = [re.sub(r"(?<!\w)90d\b", f"{gap}d", re.sub(r"(?<!\w)30d\b", f"{recent}d", value))
                  .replace("times 30 /", f"times {recent} /") for value in DEFINITIONS[name]]
        lines.append("| `" + name + "` | " + " | ".join(values) + " |")
    (Path(out) / "feature_dictionary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def feature_quality(x, train_indices):
    """Summarize training coverage without consulting held-out values."""
    train = x.iloc[train_indices].replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"feature": train.columns, "train_rows": len(train),
                         "missing_fraction": train.isna().mean().to_numpy(),
                         "unique_nonmissing": train.nunique().to_numpy(),
                         "used_by_trees": train.notna().any().to_numpy()})
