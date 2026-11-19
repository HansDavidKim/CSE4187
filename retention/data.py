"""Read-only input adapters and explicit schema validation."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np

import duckdb
import pandas as pd


SCHEMA = {
    "customers": ["customer_id", "signed_up_at"],
    "sessions": ["session_id", "customer_id", "started_at", "ended_at"],
    "events": ["event_id", "customer_id", "session_id", "event_at", "event_type"],
    "orders": ["order_id", "customer_id", "session_id", "purchased_at"],
    "items": ["order_id", "product_id", "unit_price", "quantity"],
    "coupons": ["coupon_id", "customer_id", "policy"],
    "redemptions": ["coupon_id", "order_id"],
}
TIME_COLS = {"signed_up_at", "started_at", "ended_at", "event_at", "purchased_at", "sent_at", "expires_at"}
ALIASES = {"page_view": "view", "add_to_cart": "cart", "remove": "remove_from_cart"}


def timestamp(value):
    return pd.Timestamp(value).tz_localize("UTC").tz_localize(None) if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC").tz_localize(None)


@dataclass
class Records:
    tables: dict
    metadata: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.tables[key]


def normalize(frame):
    frame = frame.copy()
    for col in frame:
        if col.endswith("_id"):
            frame[col] = frame[col].astype("string")
        elif col in TIME_COLS:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="raise", format="mixed").dt.tz_localize(None)
    if "event_type" in frame:
        frame["event_type"] = frame.event_type.replace(ALIASES)
    return frame


def validate(records):
    tables = records.tables
    counts = {}
    for name, columns in SCHEMA.items():
        frame = tables[name]
        missing = set(columns) - set(frame)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")
        counts[name] = len(frame)
        nullable = {"session_id", "ended_at"}
        required = [x for x in columns if x not in nullable]
        if frame[required].isna().any().any():
            raise ValueError(f"{name}: null in required fields")
    for name, keys in {"customers": ["customer_id"], "sessions": ["customer_id", "session_id"],
                       "events": ["event_id"], "orders": ["order_id"], "coupons": ["coupon_id"],
                       "redemptions": ["coupon_id", "order_id"]}.items():
        frame = tables[name].dropna(subset=keys)
        if frame.duplicated(keys).any():
            raise ValueError(f"{name}: duplicate key {keys}")
    customers = set(tables["customers"].customer_id)
    for name in ("sessions", "events", "orders", "coupons"):
        if not set(tables[name].customer_id).issubset(customers):
            raise ValueError(f"{name}: unknown customer")
    sessions = tables["sessions"]
    if (sessions.ended_at < sessions.started_at).any():
        raise ValueError("sessions: end before start")
    pairs = set(map(tuple, sessions[["customer_id", "session_id"]].dropna().to_numpy()))
    for name in ("events", "orders"):
        frame = tables[name].dropna(subset=["session_id"])
        if not set(map(tuple, frame[["customer_id", "session_id"]].to_numpy())).issubset(pairs):
            raise ValueError(f"{name}: unknown customer/session pair")
    if not set(tables["events"].event_type).issubset({"view", "cart", "remove_from_cart", "purchase"}):
        raise ValueError("events: unsupported event type")
    items = tables["items"]
    for col in ("unit_price", "quantity"):
        items[col] = pd.to_numeric(items[col], errors="raise")
    if ((items.unit_price < 0) | (items.quantity <= 0)).any():
        raise ValueError("items: negative price or non-positive quantity")
    if not np.isfinite(items[["unit_price", "quantity"]].to_numpy(dtype=float)).all():
        raise ValueError("items: non-finite price or quantity")
    order_ids = set(tables["orders"].order_id)
    if not set(items.order_id).issubset(order_ids):
        raise ValueError("items: unknown order")
    if not order_ids.issubset(set(items.order_id)):
        raise ValueError("orders: missing item records")
    redeemed = tables["redemptions"]
    if not set(redeemed.order_id).issubset(order_ids) or not set(redeemed.coupon_id).issubset(set(tables["coupons"].coupon_id)):
        raise ValueError("redemptions: unknown coupon or order")
    if len(redeemed):
        joined = redeemed.merge(tables["orders"][["order_id", "customer_id", "purchased_at"]], on="order_id").merge(tables["coupons"], on="coupon_id", suffixes=("_order", "_coupon"))
        if (joined.customer_id_order != joined.customer_id_coupon).any():
            raise ValueError("redemptions: coupon owner differs from order customer")
        if "sent_at" in joined and (joined.purchased_at < joined.sent_at).any():
            raise ValueError("redemptions: used before send")
        if "expires_at" in joined and (joined.purchased_at >= joined.expires_at).any():
            raise ValueError("redemptions: used after expiration")
    coupons = tables["coupons"]
    for col in ("sent_at", "expires_at"):
        if col in coupons and coupons[col].isna().any():
            raise ValueError(f"coupons: optional {col} must be complete if provided; otherwise omit column")
    if {"sent_at", "expires_at"}.issubset(coupons) and (coupons.expires_at < coupons.sent_at).any():
        raise ValueError("coupons: expiration before send")
    return {"counts": counts, "missing_session_ids": {n: int(tables[n].session_id.isna().sum()) for n in ("sessions", "events", "orders")}}


def load_canonical(directory):
    directory = Path(directory)
    tables, inputs = {}, []
    for name in SCHEMA:
        candidates = [directory / f"{name}.{suffix}" for suffix in ("parquet", "csv")]
        paths = [p for p in candidates if p.is_file()]
        if len(paths) != 1:
            raise ValueError(f"Expected exactly one {name}.csv or {name}.parquet")
        p = paths[0]
        # Project only contract columns; reject extra latent simulator fields as model inputs.
        con = duckdb.connect()
        con.execute("SET threads=2")
        con.execute("SET memory_limit='700MB'")
        try:
            relation = con.read_parquet(str(p)) if p.suffix == ".parquet" else con.read_csv(str(p), all_varchar=True)
            missing = set(SCHEMA[name]) - set(relation.columns)
            if missing:
                raise ValueError(f"{name}: missing columns {sorted(missing)}")
            wanted = SCHEMA[name] + (["sent_at", "expires_at"] if name == "coupons" else [])
            present = [c for c in wanted if c in relation.columns]
            frame = relation.project(", ".join(f'"{c}"' for c in present)).df()
        finally:
            con.close()
        tables[name] = normalize(frame)
        with p.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inputs.append({"path": str(p.resolve()), "bytes": p.stat().st_size,
                       "sha256": digest})
    records = Records(tables, {"mode": "canonical", "inputs": inputs})
    records.metadata["validation"] = validate(records)
    return records


