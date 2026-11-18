"""Point-in-time features and order-invariant timestamp-group sequences."""
import numpy as np
import pandas as pd

ACTIONS = ("view", "cart", "remove_from_cart", "purchase")
DAY = pd.Timedelta(days=1)


def safe_ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def sequences(events, as_of, max_length=128):
    """Counts per timestamp -> learned embedding; ties never use file row order."""
    if events.empty:
        return np.zeros((1, 8), dtype=np.float32), 0
    grouped = events.groupby("event_at", sort=True)
    counts = pd.crosstab(events.event_at, events.event_type).reindex(columns=ACTIONS, fill_value=0)
    times = counts.index
    dt = pd.Series(times, index=times).diff().dt.total_seconds().fillna(0).to_numpy() / 3600
    # Boundaries depend only on adjacent groups' session-ID sets, never row ordering.
    ids = [set(g.session_id.dropna()) for _, g in grouped]
    boundary = [0.] + [float(ids[i] != ids[i - 1]) for i in range(1, len(ids))]
    unknown = [float(g.session_id.isna().any()) for _, g in grouped]
    since = np.full(len(counts), np.log1p((as_of - times[-1]).total_seconds() / 3600))
    seq = np.column_stack([np.log1p(counts.to_numpy()), np.log1p(dt), boundary, unknown, since])
    return seq[-max_length:].astype(np.float32), max(0, len(seq) - max_length)


class FeatureBuilder:
    def __init__(self, records, lookback_days=30, max_length=128):
        self.records, self.lookback_days, self.max_length = records, lookback_days, max_length
        self.groups = {name: {str(u): g for u, g in frame.groupby("customer_id", sort=False)}
                       for name, frame in records.tables.items() if "customer_id" in frame}
        orders = records["orders"].copy()
        items = records["items"].copy()
        items["amount"] = pd.to_numeric(items.unit_price) * pd.to_numeric(items.quantity)
        self.products = items.groupby("order_id").product_id.agg(set).to_dict()
        sums = items.groupby("order_id").amount.sum()
        orders["amount"] = orders.order_id.map(sums)
        self.orders = {str(u): g for u, g in orders.groupby("customer_id", sort=False)}
        red = records["redemptions"].merge(orders[["order_id", "customer_id", "purchased_at"]], on="order_id")
        self.redemptions = {str(u): g for u, g in red.groupby("customer_id", sort=False)}

    def group(self, name, user):
        return self.groups[name].get(str(user), self.records[name].iloc[:0])

    def build(self, user, as_of, signup):
        start = as_of - self.lookback_days * DAY
        all_e = self.group("events", user)
        all_s = self.group("sessions", user)
        all_o = self.orders.get(str(user), self.records["orders"].assign(amount=np.nan).iloc[:0])
        past_e = all_e[all_e.event_at < as_of]
        past_s = all_s[all_s.started_at < as_of]
        past_o = all_o[all_o.purchased_at < as_of]
        e = past_e[past_e.event_at >= start]
        s = past_s[past_s.started_at >= start]
        o = past_o[past_o.purchased_at >= start]
        completed = s[s.ended_at.notna() & (s.ended_at < as_of)]
        known_orders = self.records.metadata.get("mode") == "canonical"
        purchase_times = past_o.purchased_at if known_orders else past_e.loc[past_e.event_type.eq("purchase"), "event_at"]
        def recency(times):
            return (as_of - times.max()) / DAY if len(times) else np.nan
        f = {"visit_recency_days": recency(past_s.started_at), "purchase_recency_days": recency(purchase_times),
             "age_days": (as_of - signup) / DAY if pd.notna(signup) else np.nan,
             "has_purchase_history": float(len(purchase_times) > 0),
             "sessions_30d": len(s), "active_days_30d": e.event_at.dt.floor("D").nunique()}
        for action in ACTIONS:
            f[f"{action}_count_30d"] = int(e.event_type.eq(action).sum())
        f.update(orders_30d=len(o) if known_orders else np.nan,
                 gross_spend_30d=o.amount.sum() if known_orders else np.nan,
                 mean_order_value_30d=o.amount.mean() if known_orders else np.nan,
                 duration_minutes_30d=((completed.ended_at - completed.started_at).dt.total_seconds() / 60).mean())
        ce = e[e.session_id.isin(completed.session_id.dropna())]
        purchased = set(ce.loc[ce.event_type.eq("purchase"), "session_id"].dropna())
        carts = set(ce.loc[ce.event_type.eq("cart"), "session_id"].dropna())
        f["views_per_session_30d"] = safe_ratio(ce.event_type.eq("view").sum(), len(completed))
        f["purchase_session_rate_30d"] = safe_ratio(len(purchased), len(completed))
        f["cart_session_conversion_30d"] = safe_ratio(len(carts & purchased), len(carts))
        gaps = purchase_times.sort_values().diff().dt.total_seconds().dropna() / 86400
        f["mean_purchase_gap_days"] = gaps.mean()
        f["has_purchase_gap"] = float(len(gaps) > 0)
        f["purchase_cycle_anomaly"] = safe_ratio(f["purchase_recency_days"], gaps.mean())
        f["weekend_activity_ratio"] = safe_ratio(e.event_at.dt.dayofweek.ge(5).sum(), len(e))
        f["night_activity_ratio"] = safe_ratio(e.event_at.dt.hour.lt(6).sum(), len(e))
        for side, lo, hi in (("recent", as_of - 14 * DAY, as_of), ("previous", as_of - 28 * DAY, as_of - 14 * DAY)):
            ee = past_e[(past_e.event_at >= lo) & (past_e.event_at < hi)]
            ss = past_s[(past_s.started_at >= lo) & (past_s.started_at < hi)]
            oo = past_o[(past_o.purchased_at >= lo) & (past_o.purchased_at < hi)]
            for key, value in {"sessions": len(ss), "views": ee.event_type.eq("view").sum(),
                               "carts": ee.event_type.eq("cart").sum(),
                               "orders": len(oo) if known_orders else np.nan,
                               "spend": oo.amount.sum() if known_orders else np.nan}.items():
                f[f"{key}_{side}_14d"] = value
        for key in ("sessions", "views", "carts", "orders", "spend"):
            prev, recent = f[f"{key}_previous_14d"], f[f"{key}_recent_14d"]
            f[f"{key}_change"] = safe_ratio(recent - prev, prev)
            f[f"{key}_previous_zero"] = float(prev == 0) if pd.notna(prev) else np.nan
        if known_orders:
            n = len(past_o)
            f["journey_stage"] = min(n, 3)
            transition = signup if n == 0 else past_o.purchased_at.sort_values().iloc[min(n, 3) - 1]
            f["stage_tenure_days"] = (as_of - transition) / DAY
        else:
            f["journey_stage"] = f["stage_tenure_days"] = np.nan
        grouped = pd.crosstab(e.event_at, e.event_type).reindex(columns=ACTIONS, fill_value=0)
        signatures = [tuple(row) for row in (grouped.to_numpy() > 0)]
        f["action_group_transitions"] = sum(a != b for a, b in zip(signatures, signatures[1:]))
        run = 0
        for signature in reversed(signatures):
            if signature != signatures[-1]:
                break
            run += 1
        f["last_action_group_run"] = run
        red = self.redemptions.get(str(user))
        f["coupon_uses_30d"] = (int(((red.purchased_at >= start) & (red.purchased_at < as_of)).sum()) if red is not None else 0) if known_orders else np.nan
        coupons = self.group("coupons", user)
        f["coupon_sends_30d"] = f["coupon_send_recency_days"] = f["valid_unused_coupons"] = np.nan
        if "sent_at" in coupons and coupons.sent_at.notna().all():
            sent = coupons[coupons.sent_at < as_of]
            f["coupon_sends_30d"] = int(sent.sent_at.ge(start).sum())
            f["coupon_send_recency_days"] = recency(sent.sent_at)
            if "expires_at" in sent and sent.expires_at.notna().all():
                used = set(red.loc[red.purchased_at < as_of, "coupon_id"]) if red is not None else set()
                f["valid_unused_coupons"] = int((sent.expires_at.gt(as_of) & ~sent.coupon_id.isin(used)).sum())
        if known_orders:
            products = set()
            for order_id in o.order_id:
                products.update(self.products.get(order_id, ()))
            f["unique_purchase_products_30d"] = len(products)
        else:
            f["unique_purchase_products_30d"] = np.nan
        sequence, truncated = sequences(e, as_of, self.max_length)
        return f, sequence, truncated
