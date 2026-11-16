"""Validated, immutable SQLite datasets for the seven KIT business relations."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any

import pandas as pd


RAW_COLUMNS = {
    "customers": ("customer_id", "signed_up_at"),
    "products": ("product_id", "category_id", "unit_price_minor"),
    "sessions": ("session_id", "customer_id", "started_at", "ended_at"),
    "events": ("event_id", "customer_id", "session_id", "event_at", "event_type", "product_id",
               "quantity", "unit_price_minor", "offer_product_id", "offer_unit_price_minor",
               "related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor",
               "rating", "rating_kind", "contact_id"),
    "orders": ("order_id", "purchase_event_id", "checkout_source", "coupon_id", "gross_minor",
               "discount_minor", "paid_minor"),
    "order_items": ("order_id", "line_no", "product_id", "quantity", "unit_price_minor"),
    "coupons": ("coupon_id", "customer_id", "sent_at", "expires_at", "discount_rate", "policy_id", "send_cost_minor"),
}
_TIMES = {"signed_up_at", "started_at", "ended_at", "event_at", "sent_at", "expires_at"}
_INTS = {"event_id", "quantity", "unit_price_minor", "offer_unit_price_minor", "purchase_event_id",
         "gross_minor", "discount_minor", "paid_minor", "line_no", "send_cost_minor",
         "related_purchase_event_id", "order_line_no", "rating"}
_USAGE_FIELDS = ("related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor")
_AUX_FIELDS = ("rating", "rating_kind", "contact_id")
_NULLABLE = {("products", "category_id"), ("sessions", "ended_at"), ("orders", "coupon_id"),
             *(("events", name) for name in ("session_id", "product_id", "quantity", "unit_price_minor",
                                              "offer_product_id", "offer_unit_price_minor", *_USAGE_FIELDS, *_AUX_FIELDS))}
_OPTIONAL = {"offer_product_id", "offer_unit_price_minor", *_USAGE_FIELDS, *_AUX_FIELDS}
_SCHEMA = """
CREATE TABLE customers(customer_id TEXT PRIMARY KEY NOT NULL, signed_up_at TEXT NOT NULL);
CREATE TABLE products(product_id TEXT PRIMARY KEY NOT NULL, category_id TEXT,
 unit_price_minor INTEGER NOT NULL CHECK(unit_price_minor >= 0));
CREATE TABLE sessions(session_id TEXT PRIMARY KEY NOT NULL, customer_id TEXT NOT NULL REFERENCES customers,
 started_at TEXT NOT NULL, ended_at TEXT, UNIQUE(session_id,customer_id));
CREATE TABLE events(event_id INTEGER PRIMARY KEY CHECK(event_id > 0), customer_id TEXT NOT NULL REFERENCES customers,
 session_id TEXT NOT NULL, event_at TEXT NOT NULL,
 event_type TEXT NOT NULL CHECK(event_type IN ('view','cart','remove_from_cart','purchase')),
 product_id TEXT REFERENCES products, quantity INTEGER CHECK(quantity > 0),
 unit_price_minor INTEGER CHECK(unit_price_minor >= 0), offer_product_id TEXT REFERENCES products,
 offer_unit_price_minor INTEGER CHECK(offer_unit_price_minor >= 0),
 FOREIGN KEY(session_id,customer_id) REFERENCES sessions(session_id,customer_id));
CREATE TABLE coupons(coupon_id TEXT PRIMARY KEY NOT NULL, customer_id TEXT NOT NULL REFERENCES customers,
 sent_at TEXT NOT NULL, expires_at TEXT NOT NULL, discount_rate REAL NOT NULL CHECK(discount_rate BETWEEN 0 AND 1),
 policy_id TEXT NOT NULL, send_cost_minor INTEGER NOT NULL CHECK(send_cost_minor >= 0));
CREATE TABLE orders(order_id TEXT PRIMARY KEY NOT NULL,
 purchase_event_id INTEGER UNIQUE NOT NULL REFERENCES events,
 checkout_source TEXT NOT NULL CHECK(checkout_source IN ('cart','buy_now')),
 coupon_id TEXT UNIQUE REFERENCES coupons, gross_minor INTEGER NOT NULL CHECK(gross_minor >= 0),
 discount_minor INTEGER NOT NULL CHECK(discount_minor >= 0 AND discount_minor <= gross_minor),
 paid_minor INTEGER NOT NULL CHECK(paid_minor = gross_minor-discount_minor));
CREATE TABLE order_items(order_id TEXT NOT NULL REFERENCES orders, line_no INTEGER NOT NULL CHECK(line_no > 0),
 product_id TEXT NOT NULL REFERENCES products, quantity INTEGER NOT NULL CHECK(quantity > 0),
 unit_price_minor INTEGER NOT NULL CHECK(unit_price_minor >= 0), PRIMARY KEY(order_id,line_no));
CREATE INDEX event_customer_time ON events(customer_id,event_at,event_id);
CREATE INDEX session_customer_time ON sessions(customer_id,started_at);
CREATE INDEX coupon_customer_time ON coupons(customer_id,sent_at);
PRAGMA user_version=1;
"""


def _missing(value: Any) -> bool:
    return value is None or value is pd.NA or bool(pd.isna(value))


def _integer(value: Any, name: str) -> int:
    # Nullable pandas columns can represent exact integer values as floats.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, numbers.Integral) and (
        not math.isfinite(value) or not float(value).is_integer() or abs(value) > 2 ** 53
    ):
        raise ValueError(f"{name} must be an exact integer")
    result = int(value)
    if not -(2 ** 63) <= result < 2 ** 63:
        raise ValueError(f"{name} exceeds SQLite integer range")
    return result


def _utc(value: Any, name: str = "timestamp") -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a UTC-aware timestamp") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name} must be a UTC-aware timestamp")
    return result.tz_convert("UTC")


def _rate(value: Any) -> Decimal:
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("discount_rate must be between zero and one") from exc
    if isinstance(value, bool) or not rate.is_finite() or not 0 <= rate <= 1:
        raise ValueError("discount_rate must be between zero and one")
    return rate


def discount_minor(gross_minor: int, discount_rate: float) -> int:
    """Round a base amount's discount half up in its declared minor unit."""
    gross = _integer(gross_minor, "gross_minor")
    if gross < 0:
        raise ValueError("gross_minor must be nonnegative")
    return int((Decimal(gross) * _rate(discount_rate)).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _canonical(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(tables) != set(RAW_COLUMNS):
        raise ValueError(f"raw dataset requires exactly these tables: {list(RAW_COLUMNS)}")
    result = {}
    for name, columns in RAW_COLUMNS.items():
        frame = tables[name]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{name} must be a DataFrame")
        extra = set(frame) - set(columns)
        missing = set(columns) - set(frame) - (_OPTIONAL if name == "events" else set())
        if extra or (missing and not frame.empty):
            raise ValueError(f"{name} columns: missing={sorted(missing)}, unexpected={sorted(extra)}")
        frame = frame.reindex(columns=columns).copy()
        for column in columns:
            values = []
            for value in frame[column]:
                if _missing(value):
                    if (name, column) not in _NULLABLE:
                        raise ValueError(f"{name}.{column} cannot be null")
                    values.append(None)
                elif column in _TIMES:
                    values.append(_utc(value, f"{name}.{column}"))
                elif column in _INTS:
                    values.append(_integer(value, f"{name}.{column}"))
                elif column == "discount_rate":
                    values.append(float(_rate(value)))
                elif isinstance(value, str) and value.strip():
                    values.append(value)
                else:
                    raise ValueError(f"{name}.{column} must be a nonempty string")
            if column in _TIMES:
                frame[column] = pd.Series(values, index=frame.index, dtype="datetime64[ns, UTC]")
            elif column in _INTS:
                frame[column] = pd.Series(values, index=frame.index, dtype="Int64")
            else:
                frame[column] = values
        result[name] = frame
    return result


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{key: None if _missing(value) else value for key, value in row.items()}
            for row in frame.to_dict("records")]


def _event_records(frame: pd.DataFrame):
    for values in frame.itertuples(index=False, name=None):
        yield {key: None if _missing(value) else value for key, value in zip(frame.columns, values)}


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_version(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value not in (1, 2, 3, 4, 5):
        raise ValueError("unsupported raw schema version")
    return int(value)


def _columns(name: str, version: int) -> tuple[str, ...]:
    if name != "events" or version >= 4:
        return RAW_COLUMNS[name]
    return RAW_COLUMNS[name][:-len(_AUX_FIELDS)] if version == 3 else RAW_COLUMNS[name][:-len(_USAGE_FIELDS)-len(_AUX_FIELDS)]


def _schema(version: int) -> str:
    schema = _SCHEMA.replace("PRAGMA user_version=1;", f"PRAGMA user_version={version};")
    if version >= 2:
        schema = schema.replace("('cart','buy_now')", "('cart','buy_now','mixed')")
    if version >= 3:
        schema = schema.replace("('view','cart','remove_from_cart','purchase')",
                                "('view','cart','remove_from_cart','purchase','coupon_usage')")
        schema = schema.replace(" FOREIGN KEY(session_id,customer_id)", """
 related_purchase_event_id INTEGER UNIQUE REFERENCES events(event_id) DEFERRABLE INITIALLY DEFERRED,
 coupon_id TEXT UNIQUE REFERENCES coupons(coupon_id) DEFERRABLE INITIALLY DEFERRED,
 order_line_no INTEGER CHECK(order_line_no > 0), discount_minor INTEGER CHECK(discount_minor >= 0),
 CHECK(related_purchase_event_id < event_id),
 CHECK((event_type='coupon_usage' AND related_purchase_event_id IS NOT NULL AND coupon_id IS NOT NULL
        AND order_line_no IS NOT NULL AND discount_minor IS NOT NULL AND quantity IS NOT NULL AND quantity=1
        AND product_id IS NOT NULL AND unit_price_minor IS NOT NULL)
       OR (event_type!='coupon_usage' AND related_purchase_event_id IS NULL AND coupon_id IS NULL
        AND order_line_no IS NULL AND discount_minor IS NULL)),
 FOREIGN KEY(session_id,customer_id)""")
    if version >= 4:
        # Partial indexes avoid indexing millions of NULL accounting references.
        schema = _SCHEMA.replace("PRAGMA user_version=1;", f"PRAGMA user_version={version};")
        schema = schema.replace("('cart','buy_now')", "('cart','buy_now','mixed')")
        schema = schema.replace("session_id TEXT NOT NULL, event_at", "session_id TEXT, event_at")
        schema = schema.replace("('view','cart','remove_from_cart','purchase')",
            "('view','cart','remove_from_cart','purchase','coupon_usage','search','review','cs_contact')")
        schema = schema.replace(" FOREIGN KEY(session_id,customer_id)", """
 related_purchase_event_id INTEGER REFERENCES events(event_id) DEFERRABLE INITIALLY DEFERRED,
 coupon_id TEXT REFERENCES coupons(coupon_id) DEFERRABLE INITIALLY DEFERRED,
 order_line_no INTEGER CHECK(order_line_no > 0), discount_minor INTEGER CHECK(discount_minor >= 0),
 rating INTEGER CHECK(rating BETWEEN 1 AND 5), rating_kind TEXT, contact_id TEXT,
 CHECK(related_purchase_event_id < event_id),
 CHECK((event_type='coupon_usage' AND related_purchase_event_id IS NOT NULL AND coupon_id IS NOT NULL
        AND order_line_no IS NOT NULL AND discount_minor IS NOT NULL)
       OR (event_type='review' AND related_purchase_event_id IS NOT NULL AND coupon_id IS NULL
           AND order_line_no IS NULL AND discount_minor IS NULL)
       OR (event_type NOT IN ('coupon_usage','review') AND related_purchase_event_id IS NULL
           AND coupon_id IS NULL AND order_line_no IS NULL AND discount_minor IS NULL)),
 CHECK((event_type='review' AND rating IS NOT NULL AND rating_kind IS NOT NULL
        AND rating_kind IN ('product_star_scenario','order_survey_score')
        AND contact_id IS NULL AND session_id IS NULL)
       OR (event_type='cs_contact' AND rating IS NULL AND rating_kind IS NULL
           AND contact_id IS NOT NULL AND session_id IS NULL)
       OR (event_type NOT IN ('review','cs_contact') AND rating IS NULL AND rating_kind IS NULL
           AND contact_id IS NULL AND session_id IS NOT NULL)),
 FOREIGN KEY(session_id,customer_id)""")
        schema += """
CREATE UNIQUE INDEX coupon_usage_purchase ON events(related_purchase_event_id) WHERE event_type='coupon_usage';
CREATE UNIQUE INDEX coupon_usage_coupon ON events(coupon_id) WHERE event_type='coupon_usage';
CREATE UNIQUE INDEX review_purchase ON events(related_purchase_event_id) WHERE event_type='review';
CREATE UNIQUE INDEX cs_contact_identifier ON events(contact_id) WHERE event_type='cs_contact';
"""
        if version >= 5:
            schema = schema.replace("AND order_line_no IS NULL AND discount_minor IS NULL)",
                                    "AND order_line_no IS NOT NULL AND discount_minor IS NULL)", 1)
            schema = schema.replace("AND contact_id IS NULL AND session_id IS NULL)",
                """AND contact_id IS NULL AND session_id IS NULL AND product_id IS NOT NULL
        AND order_line_no IS NOT NULL AND quantity IS NULL AND unit_price_minor IS NULL
        AND offer_product_id IS NULL AND offer_unit_price_minor IS NULL)""", 1)
            schema = schema.replace("CREATE UNIQUE INDEX review_purchase ON events(related_purchase_event_id)",
                                    "CREATE UNIQUE INDEX review_purchase_product ON events(related_purchase_event_id,product_id)")
    return schema


def _validate(tables: dict[str, pd.DataFrame], metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    version = _schema_version(metadata.get("raw_schema_version", 1))
    for key in ("dataset_id", "currency"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be explicit")
    _integer(metadata.get("seed"), "metadata.seed")
    try:
        unit = Decimal(str(metadata.get("amount_unit")))
        if not unit.is_finite() or unit <= 0:
            raise ValueError
        json.dumps(metadata, allow_nan=False)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("metadata requires a positive amount_unit and JSON-serializable values") from exc
    rows = {name: _records(frame) for name, frame in tables.items() if name != "events"}
    keys = {"customers": ("customer_id",), "products": ("product_id",), "sessions": ("session_id",),
            "events": ("event_id",), "orders": ("order_id",), "order_items": ("order_id", "line_no"),
            "coupons": ("coupon_id",)}
    for name, columns in keys.items():
        if tables[name].duplicated(subset=list(columns)).any():
            raise ValueError(f"duplicate {name} primary key")
    customers = {row["customer_id"]: row for row in rows["customers"]}
    products = {row["product_id"]: row for row in rows["products"]}
    sessions = {row["session_id"]: row for row in rows["sessions"]}
    ordered_events = tables["events"].sort_values("event_id")
    purchase_events = {}
    usage_events = {}
    usage_coupons = set()
    review_purchases, product_reviews, contact_ids = set(), [], set()
    coupons = {row["coupon_id"]: row for row in rows["coupons"]}
    orders = {row["order_id"]: row for row in rows["orders"]}
    for product in products.values():
        if product["unit_price_minor"] < 0:
            raise ValueError("product price must be nonnegative")
    for session in sessions.values():
        customer = customers.get(session["customer_id"])
        if customer is None or session["started_at"] < customer["signed_up_at"]:
            raise ValueError("session must belong to a signed-up customer")
        if session["ended_at"] is not None and session["ended_at"] < session["started_at"]:
            raise ValueError("session end precedes start")
    for coupon in coupons.values():
        customer = customers.get(coupon["customer_id"])
        if (customer is None or coupon["sent_at"] < customer["signed_up_at"] or
                coupon["expires_at"] <= coupon["sent_at"] or coupon["send_cost_minor"] < 0):
            raise ValueError("invalid coupon ownership, validity or cost")
    previous_at = None
    for event in _event_records(ordered_events):
        kind = event["event_type"]
        asynchronous = version >= 4 and kind in {"review", "cs_contact"}
        session = sessions.get(event["session_id"])
        customer = customers.get(event["customer_id"])
        if event["event_id"] <= 0 or customer is None or event["event_at"] < customer["signed_up_at"]:
            raise ValueError("event must belong to a signed-up customer")
        if asynchronous:
            if event["session_id"] is not None:
                raise ValueError("asynchronous review and CS events must not fabricate shopping sessions")
            if metadata.get("observation_end_at") and event["event_at"] >= _utc(metadata["observation_end_at"]):
                raise ValueError("asynchronous event is outside observation")
        elif (session is None or event["customer_id"] != session["customer_id"] or
                event["event_at"] < session["started_at"] or
                (session["ended_at"] is not None and event["event_at"] > session["ended_at"])):
            raise ValueError("invalid event ID, session owner or time")
        if previous_at is not None and event["event_at"] < previous_at:
            raise ValueError("event IDs must follow event time order")
        previous_at = event["event_at"]
        allowed = {"view", "cart", "remove_from_cart", "purchase"}
        if version >= 3:
            allowed.add("coupon_usage")
        if version >= 4:
            allowed.update(("search", "review", "cs_contact"))
        if kind not in allowed:
            raise ValueError("invalid event type")
        if kind not in {"coupon_usage", "review"} and any(event[key] is not None for key in _USAGE_FIELDS):
            raise ValueError("coupon usage fields must be null on ordinary events")
        if kind not in {"review", "cs_contact"} and any(event[key] is not None for key in _AUX_FIELDS):
            raise ValueError("review/CS fields must be null on other events")
        if kind in {"search", "review", "cs_contact"}:
            catalog_fields = ("quantity", "unit_price_minor", "offer_product_id", "offer_unit_price_minor")
            if not (version >= 5 and kind == "review"):
                catalog_fields = ("product_id", *catalog_fields)
            if any(event[key] is not None for key in catalog_fields):
                raise ValueError("auxiliary events cannot invent catalog interactions")
            if kind == "search":
                continue
            if kind == "review":
                related = event["related_purchase_event_id"]
                purchase = purchase_events.get(related)
                review_key = (related, event["product_id"]) if version >= 5 else related
                if (purchase is None or related >= event["event_id"] or review_key in review_purchases or
                        purchase["customer_id"] != event["customer_id"] or event["event_at"] < purchase["event_at"]):
                    raise ValueError("review requires a unique earlier purchase/product of the same customer")
                forbidden_linkage = ("coupon_id", "discount_minor") if version >= 5 else _USAGE_FIELDS[1:]
                if (event["rating"] not in (1, 2, 3, 4, 5) or event["rating_kind"] not in
                        {"product_star_scenario", "order_survey_score"} or event["contact_id"] is not None or
                        any(event[key] is not None for key in forbidden_linkage)):
                    raise ValueError("invalid review score or order linkage")
                if version >= 5:
                    if event["product_id"] not in products or event["order_line_no"] is None or event["order_line_no"] <= 0:
                        raise ValueError("product review requires a catalog product and positive purchased line")
                    product_reviews.append({key: event[key] for key in ("related_purchase_event_id", "product_id", "order_line_no")})
                review_purchases.add(review_key)
            elif (event["rating"] is not None or event["rating_kind"] is not None or
                  not event["contact_id"] or event["contact_id"] in contact_ids):
                raise ValueError("CS contacts require unique contact IDs and no rating")
            else:
                contact_ids.add(event["contact_id"])
            continue
        if kind == "coupon_usage":
            if (any(event[key] is None for key in _USAGE_FIELDS) or event["quantity"] != 1
                    or event["order_line_no"] <= 0 or event["discount_minor"] < 0):
                raise ValueError("coupon usage requires purchase, coupon, positive line, one unit and nonnegative discount")
            related = event["related_purchase_event_id"]
            purchase = purchase_events.get(related)
            if (purchase is None or related >= event["event_id"] or
                    any(event[key] != purchase[key] for key in ("customer_id", "session_id", "event_at"))):
                raise ValueError("coupon usage must follow its purchase in the same customer, session and time")
            if related in usage_events or event["coupon_id"] in usage_coupons:
                raise ValueError("each coupon and purchase permits exactly one coupon usage")
            usage_events[related] = event
            usage_coupons.add(event["coupon_id"])
        if kind == "purchase":
            if any(event[key] is not None for key in ("product_id", "quantity", "unit_price_minor")):
                raise ValueError("purchase composition belongs in order_items")
            purchase_events[event["event_id"]] = {key: event[key] for key in ("customer_id", "session_id", "event_at", "offer_product_id")}
        else:
            product = products.get(event["product_id"])
            if product is None or event["unit_price_minor"] != product["unit_price_minor"]:
                raise ValueError("event requires a catalog product and its fixed display price")
            if ((kind == "view" and event["quantity"] is not None) or
                    (kind == "cart" and event["quantity"] != 1) or
                    (kind == "remove_from_cart" and (event["quantity"] is None or event["quantity"] <= 0))):
                raise ValueError("invalid event quantity; cart adds one item")
        offered = event["offer_product_id"]
        if offered is None:
            if event["offer_unit_price_minor"] is not None:
                raise ValueError("offer price requires offer product")
        elif offered not in products or event["offer_unit_price_minor"] != products[offered]["unit_price_minor"]:
            raise ValueError("offer requires its fixed catalog price")
    items: dict[str, list[dict]] = {order_id: [] for order_id in orders}
    for item in rows["order_items"]:
        product = products.get(item["product_id"])
        if (item["order_id"] not in orders or product is None or item["line_no"] <= 0 or
                item["quantity"] <= 0 or item["unit_price_minor"] != product["unit_price_minor"]):
            raise ValueError("invalid order item reference, quantity or fixed price")
        items[item["order_id"]].append(item)
    purchases: dict[int, dict] = {}
    used_coupons: set[str] = set()
    for order in orders.values():
        event = purchase_events.get(order["purchase_event_id"])
        if event is None or order["purchase_event_id"] in purchases:
            raise ValueError("each order requires a unique purchase event")
        sources = {"cart", "buy_now"} if version == 1 else {"cart", "buy_now", "mixed"}
        if order["checkout_source"] not in sources or not items[order["order_id"]]:
            raise ValueError("order requires checkout_source and at least one item")
        gross = sum(item["quantity"] * item["unit_price_minor"] for item in items[order["order_id"]])
        coupon_id = order["coupon_id"]
        usage = usage_events.get(order["purchase_event_id"])
        if version >= 3 and ((coupon_id is None) != (usage is None)):
            raise ValueError("raw schema version 3 requires exactly one usage for every coupon order and none for other orders")
        discount = 0
        if coupon_id is not None:
            coupon = coupons.get(coupon_id)
            if (coupon is None or coupon_id in used_coupons or coupon["customer_id"] != event["customer_id"] or
                    not coupon["sent_at"] <= event["event_at"] < coupon["expires_at"]):
                raise ValueError("coupon must be owned, valid and used at most once")
            if version >= 3:
                # User-requested allocation policy, not an estimated source preference.
                target = min(items[order["order_id"]], key=lambda item: (-item["unit_price_minor"], item["line_no"]))
                if usage["coupon_id"] != coupon_id:
                    raise ValueError("coupon usage coupon must match its purchase order")
                if (usage["order_line_no"] != target["line_no"] or
                        usage["product_id"] != target["product_id"] or usage["unit_price_minor"] != target["unit_price_minor"]):
                    raise ValueError("coupon usage must target the highest unit price, breaking ties by lower line number")
                discount = discount_minor(target["unit_price_minor"], coupon["discount_rate"])
                if usage["discount_minor"] != discount:
                    raise ValueError("coupon usage discount must equal one purchased unit's half-up discount")
            else:
                discount = discount_minor(gross, coupon["discount_rate"])
            used_coupons.add(coupon_id)
        if (order["gross_minor"] != gross or order["discount_minor"] != discount or
                order["paid_minor"] != gross - discount):
            raise ValueError("order gross, discount and paid must reconcile")
        purchases[order["purchase_event_id"]] = order
    if set(purchases) != set(purchase_events):
        raise ValueError("every purchase event requires exactly one order")
    for review in product_reviews:
        order = purchases[review["related_purchase_event_id"]]
        if not any(item["line_no"] == review["order_line_no"] and item["product_id"] == review["product_id"]
                   for item in items[order["order_id"]]):
            raise ValueError("product review must reference its actual purchased product and order line")
    carts: dict[str, Counter] = {customer_id: Counter() for customer_id in customers}
    for event in _event_records(ordered_events):
        cart = carts[event["customer_id"]]
        if event["event_type"] == "cart":
            cart[event["product_id"]] += event["quantity"]
        elif event["event_type"] == "remove_from_cart":
            cart[event["product_id"]] -= event["quantity"]
            if cart[event["product_id"]] < 0:
                raise ValueError("cart quantity becomes negative")
        elif event["event_type"] == "purchase":
            order = purchases[event["event_id"]]
            basket: Counter = Counter()
            for item in items[order["order_id"]]:
                basket[item["product_id"]] += item["quantity"]
            if order["checkout_source"] == "cart":
                current = +cart
                if version == 1 and (not current or basket != current):
                    raise ValueError("cart checkout must purchase the complete current cart")
                if any(quantity > cart[product] for product, quantity in basket.items()):
                    raise ValueError("cart checkout exceeds held quantity")
                cart.subtract(basket)
            elif order["checkout_source"] == "mixed":
                deduction = Counter({product: min(cart[product], quantity) for product, quantity in basket.items()})
                if not 0 < sum(deduction.values()) < sum(basket.values()):
                    raise ValueError("mixed checkout requires both held cart units and direct units")
                cart.subtract(deduction)
            elif version == 1 and sum(basket.values()) != 1:
                raise ValueError("buy_now purchases exactly one item")
            elif version == 1 and event["offer_product_id"] is not None and basket != Counter({event["offer_product_id"]: 1}):
                raise ValueError("buy_now must purchase its offered product")


def validate_raw_dataset(tables: dict[str, pd.DataFrame], metadata: dict) -> None:
    """Reject invalid business records without modifying the input frames."""
    _validate(_canonical(tables), metadata)


def cart_at(tables: dict[str, pd.DataFrame], customer_id: str, as_of_at: Any,
            before_event_id: int | None = None) -> dict[str, int]:
    """Replay prior cart changes; a supplied ID selects the same-time prefix."""
    if customer_id not in set(tables["customers"].customer_id):
        raise ValueError("unknown customer_id")
    cutoff = _utc(as_of_at, "as_of_at")
    events = tables["events"].loc[tables["events"].customer_id.eq(customer_id)].copy()
    events["event_at"] = pd.Series([_utc(value, "event_at") for value in events.event_at],
                                  index=events.index, dtype="datetime64[ns, UTC]")
    keep = events.event_at.lt(cutoff)
    if before_event_id is not None:
        before_event_id = _integer(before_event_id, "before_event_id")
        if before_event_id <= 0:
            raise ValueError("before_event_id must be positive")
        keep |= events.event_at.eq(cutoff) & events.event_id.lt(before_event_id)
    events = events.loc[keep].sort_values(["event_at", "event_id"])
    orders = tables["orders"].set_index("purchase_event_id")
    grouped = {key: frame for key, frame in tables["order_items"].groupby("order_id", sort=False)}
    cart: Counter = Counter()
    for event in events.itertuples(index=False):
        if event.event_type == "cart":
            cart[event.product_id] += _integer(event.quantity, "quantity")
        elif event.event_type == "remove_from_cart":
            cart[event.product_id] -= _integer(event.quantity, "quantity")
        elif event.event_type == "purchase":
            if event.event_id not in orders.index:
                raise ValueError("purchase event has no order")
            order = orders.loc[event.event_id]
            if order.checkout_source in {"cart", "mixed"}:
                basket: Counter = Counter()
                for item in grouped[order.order_id].itertuples(index=False):
                    basket[item.product_id] += _integer(item.quantity, "quantity")
                for product, quantity in basket.items():
                    cart[product] -= min(cart[product], quantity) if order.checkout_source == "mixed" else quantity
            elif order.checkout_source != "buy_now":
                raise ValueError("unknown checkout_source")
        if any(quantity < 0 for quantity in cart.values()):
            raise ValueError("cart quantity becomes negative")
    return dict(+cart)


def save_raw_dataset(path: str | Path, tables: dict[str, pd.DataFrame], metadata: dict) -> None:
    """Write a validated database and checksum manifest; never overwrite files."""
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    if os.path.lexists(path) or os.path.lexists(manifest_path):
        raise FileExistsError(f"raw dataset already exists: {path}")
    canonical = _canonical(tables)
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    version = _schema_version(metadata.get("raw_schema_version", 1))
    metadata = {**metadata, "raw_schema_version": version}
    _validate(canonical, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            database_tmp = Path(stream.name)
        temporary.append(database_tmp)
        connection = sqlite3.connect(database_tmp)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(_schema(version))
            with connection:
                # Load coupon parents first to avoid a full deferred-FK child scan for each coupon.
                for name in ("customers", "products", "sessions", "coupons", "events", "orders", "order_items"):
                    columns = _columns(name, version)
                    placeholders = ",".join("?" for _ in columns)
                    def values():
                        for row in canonical[name].loc[:, columns].itertuples(index=False, name=None):
                            yield tuple(None if _missing(value) else
                                        value.isoformat(timespec="nanoseconds") if column in _TIMES else
                                        int(value) if column in _INTS else value
                                        for column, value in zip(columns, row))
                    connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', values())
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise ValueError("SQLite foreign key check failed")
        finally:
            connection.close()
        digest = _checksum(database_tmp)
        manifest = {"raw_schema_version": version, "database_sha256": digest, "metadata": metadata}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".json.tmp",
                                         dir=path.parent, delete=False) as stream:
            manifest_tmp = Path(stream.name)
            temporary.append(manifest_tmp)
            json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Hard links publish without replacing a file created by another writer.
        os.link(database_tmp, path)
        published.append(path)
        os.link(manifest_tmp, manifest_path)
        published.append(manifest_path)
    except BaseException:
        for published_path in reversed(published):
            published_path.unlink()
        raise
    finally:
        for temporary_path in temporary:
            temporary_path.unlink(missing_ok=True)


def load_raw_dataset(path: str | Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Read and validate a fixed dataset, including its manifest checksum."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    version = _schema_version(manifest.get("raw_schema_version"))
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or _schema_version(metadata.get("raw_schema_version", 1)) != version:
        raise ValueError("manifest metadata raw schema version differs")
    if _checksum(path) != manifest.get("database_sha256"):
        raise ValueError("raw database checksum mismatch")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != version:
            raise ValueError("unsupported SQLite schema version")
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if names != set(RAW_COLUMNS):
            raise ValueError("raw database must contain exactly seven business tables")
        tables = {}
        for name in RAW_COLUMNS:
            columns = _columns(name, version)
            actual_columns = tuple(row[1] for row in connection.execute(f'PRAGMA table_info("{name}")'))
            if actual_columns != columns:
                raise ValueError(f"{name} columns differ from raw schema version {version}")
            integer_types = {column: "Int64" for column in columns if column in _INTS}
            tables[name] = pd.read_sql_query(f'SELECT * FROM "{name}"', connection, dtype=integer_types)
    finally:
        connection.close()
    tables = _canonical(tables)
    _validate(tables, metadata)
    return tables, metadata
