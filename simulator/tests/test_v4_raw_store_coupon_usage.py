"""Version 3 coupon allocation, explicit usage records and legacy SQLite compatibility."""
import sqlite3

import pandas as pd
import pytest

import simulate.v4.simulator.raw_store as raw
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, load_raw_dataset, save_raw_dataset, validate_raw_dataset
from test_v4_raw_store import dataset


USAGE_FIELDS = ("related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor")


@pytest.fixture
def coupon_dataset(dataset):
    tables, metadata = dataset
    metadata["raw_schema_version"] = 3
    tables["events"].loc[tables["events"].event_id.eq(8), "event_id"] = 9
    usage = dict(event_id=8, customer_id="u1", session_id="s2", event_at=pd.Timestamp("2024-01-02 10:05", tz="UTC"),
                 event_type="coupon_usage", product_id="A", quantity=1, unit_price_minor=100,
                 offer_product_id=None, offer_unit_price_minor=None, related_purchase_event_id=7,
                 coupon_id="c1", order_line_no=1, discount_minor=10)
    tables["events"] = pd.DataFrame([usage, *tables["events"].to_dict("records")], columns=RAW_COLUMNS["events"]).sort_values("event_id", ascending=False).reset_index(drop=True)
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["discount_minor", "paid_minor"]] = [10, 141]
    return tables, metadata


def test_v3_coupon_usage_roundtrip_and_stock_neutrality(coupon_dataset, tmp_path):
    tables, metadata = coupon_dataset
    original = tables["events"].copy(deep=True)
    validate_raw_dataset(tables, metadata)
    path = tmp_path / "raw.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, info = load_raw_dataset(path)
    assert info["raw_schema_version"] == 3
    assert set(loaded) == set(RAW_COLUMNS) and len(loaded) == 7
    pd.testing.assert_frame_equal(tables["events"], original)
    usage = loaded["events"].loc[loaded["events"].event_type.eq("coupon_usage")].iloc[0]
    assert (usage.related_purchase_event_id, usage.coupon_id, usage.order_line_no, usage.discount_minor) == (7, "c1", 1, 10)
    assert loaded["events"].loc[loaded["events"].event_type.ne("coupon_usage"), list(USAGE_FIELDS)].isna().all().all()
    # Purchase clears the held units; processing its later usage changes nothing.
    assert cart_at(loaded, "u1", usage.event_at, before_event_id=8) == {}
    assert cart_at(loaded, "u1", usage.event_at, before_event_id=9) == {}
    assert cart_at(loaded, "u1", "2024-01-02T10:07:00Z") == {"B": 1}
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert len(connection.execute("PRAGMA table_info(events)").fetchall()) == 14
        references = {row[3]: row[2] for row in connection.execute("PRAGMA foreign_key_list(events)")}
        assert references["related_purchase_event_id"] == "events" and references["coupon_id"] == "coupons"
        assert "'coupon_usage'" in connection.execute("SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0]
    with pytest.raises(FileExistsError):
        save_raw_dataset(path, tables, metadata)


@pytest.mark.parametrize("version", [3, 4])
def test_coupon_usage_insert_has_no_unresolved_coupon_parent(coupon_dataset, tmp_path, monkeypatch, version):
    tables, metadata = coupon_dataset
    metadata["raw_schema_version"] = version
    original_connect = sqlite3.connect
    checked = []

    class ParentFirstConnection(sqlite3.Connection):
        def executemany(self, statement, values):
            result = super().executemany(statement, values)
            if statement.startswith('INSERT INTO "events"'):
                checked.append(True)
                assert self.execute('PRAGMA foreign_key_check(events)').fetchall() == []
            return result

    monkeypatch.setattr(raw.sqlite3, "connect", lambda *args, **kwargs:
                        original_connect(*args, factory=ParentFirstConnection, **kwargs))
    save_raw_dataset(tmp_path / "raw.sqlite", tables, metadata)
    assert checked == [True]


@pytest.mark.parametrize("a_quantity,b_quantity", [(3, 1), (1, 10)])
def test_v3_one_unit_half_up_uses_unit_price_instead_of_line_or_order_total(coupon_dataset, a_quantity, b_quantity):
    tables, metadata = coupon_dataset
    tables["products"].loc[tables["products"].product_id.eq("A"), "unit_price_minor"] = 105
    events = tables["events"]
    events.loc[events.product_id.eq("A"), "unit_price_minor"] = 105
    events.loc[events.offer_product_id.eq("A"), "offer_unit_price_minor"] = 105
    events.loc[events.event_type.eq("coupon_usage"), "discount_minor"] = 11
    items = tables["order_items"]
    items.loc[items.product_id.eq("A"), "unit_price_minor"] = 105
    items.loc[items.order_id.eq("checkout") & items.product_id.eq("A"), "quantity"] = a_quantity
    items.loc[items.order_id.eq("checkout") & items.product_id.eq("B"), "quantity"] = b_quantity
    tables["orders"].loc[tables["orders"].order_id.eq("buy"), ["gross_minor", "paid_minor"]] = [105, 105]
    gross = 105 * a_quantity + 51 * b_quantity
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["checkout_source", "gross_minor", "discount_minor", "paid_minor"]] = ["buy_now", gross, 11, gross - 11]
    validate_raw_dataset(tables, metadata)
    assert cart_at(tables, "u1", "2024-01-02T10:07:00Z") == {"A": 1, "B": 2}


def test_v3_equal_unit_prices_choose_lower_line_number(coupon_dataset):
    tables, metadata = coupon_dataset
    tables["products"].loc[tables["products"].product_id.eq("B"), "unit_price_minor"] = 100
    tables["events"].loc[tables["events"].product_id.eq("B"), "unit_price_minor"] = 100
    tables["events"].loc[tables["events"].offer_product_id.eq("B"), "offer_unit_price_minor"] = 100
    tables["order_items"].loc[tables["order_items"].product_id.eq("B"), "unit_price_minor"] = 100
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["gross_minor", "paid_minor"]] = [200, 190]
    validate_raw_dataset(tables, metadata)
    tables["events"].loc[tables["events"].event_type.eq("coupon_usage"), ["product_id", "order_line_no"]] = ["B", 2]
    with pytest.raises(ValueError, match="lower line number"):
        validate_raw_dataset(tables, metadata)


def test_v3_sent_but_unused_coupon_has_no_usage_event(coupon_dataset):
    tables, metadata = coupon_dataset
    tables["events"] = tables["events"].loc[tables["events"].event_type.ne("coupon_usage")].copy()
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["coupon_id", "discount_minor", "paid_minor"]] = [None, 0, 151]
    validate_raw_dataset(tables, metadata)


@pytest.mark.parametrize("case,match", [
    ("missing", "schema version 3"), ("missing_link", "requires purchase"),
    ("ordinary_fields", "fields must be null"), ("quantity", "one unit"),
    ("related_fk", "follow its purchase"), ("owner", "session owner"),
    ("session", "same customer, session and time"), ("time", "same customer, session and time"),
    ("duplicate", "exactly one coupon usage"), ("coupon", "coupon must match"),
    ("line", "highest unit price"), ("lower_price", "highest unit price"),
    ("discount", "half-up discount"), ("whole_order_discount", "must reconcile"),
])
def test_v3_rejects_broken_usage_links_targets_and_amounts(coupon_dataset, case, match):
    tables, metadata = coupon_dataset
    events = tables["events"]
    usage = events.event_type.eq("coupon_usage")
    if case == "missing": tables["events"] = events.loc[~usage].copy()
    elif case == "missing_link": events.loc[usage, "related_purchase_event_id"] = None
    elif case == "ordinary_fields": events.loc[events.event_id.eq(2), "coupon_id"] = "c1"
    elif case == "quantity": events.loc[usage, "quantity"] = 2
    elif case == "related_fk": events.loc[usage, "related_purchase_event_id"] = 999
    elif case == "owner": events.loc[usage, "customer_id"] = "u2"
    elif case == "session":
        session = tables["sessions"].iloc[1].to_dict()
        session["session_id"] = "other"
        tables["sessions"] = pd.concat([tables["sessions"], pd.DataFrame([session])], ignore_index=True)
        events.loc[usage, "session_id"] = "other"
    elif case == "time": events.loc[usage, "event_at"] += pd.Timedelta(seconds=1)
    elif case == "duplicate":
        duplicate = events.loc[usage].iloc[0].to_dict()
        duplicate["event_id"] = 9
        events.loc[events.event_id.eq(9), "event_id"] = 10
        tables["events"] = pd.DataFrame([*events.to_dict("records"), duplicate], columns=RAW_COLUMNS["events"])
    elif case == "coupon":
        coupon = tables["coupons"].iloc[0].to_dict()
        coupon["coupon_id"] = "other"
        tables["coupons"] = pd.concat([tables["coupons"], pd.DataFrame([coupon])], ignore_index=True)
        events.loc[usage, "coupon_id"] = "other"
    elif case == "line": events.loc[usage, "order_line_no"] = 99
    elif case == "lower_price":
        events.loc[usage, ["product_id", "unit_price_minor", "order_line_no", "discount_minor"]] = ["B", 51, 2, 5]
        tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["discount_minor", "paid_minor"]] = [5, 146]
    elif case == "discount": events.loc[usage, "discount_minor"] = 11
    else: tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["discount_minor", "paid_minor"]] = [15, 136]
    with pytest.raises(ValueError, match=match):
        validate_raw_dataset(tables, metadata)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_sqlite_uses_original_columns_but_loads_optional_null_fields(dataset, tmp_path, version):
    tables, metadata = dataset
    metadata["raw_schema_version"] = version
    tables["events"] = tables["events"].drop(columns=list(USAGE_FIELDS))
    path = tmp_path / f"v{version}.sqlite"
    save_raw_dataset(path, tables, metadata)
    with sqlite3.connect(path) as connection:
        assert len(connection.execute("PRAGMA table_info(events)").fetchall()) == 10
    loaded, info = load_raw_dataset(path)
    assert info["raw_schema_version"] == version
    assert tuple(loaded["events"].columns) == RAW_COLUMNS["events"]
    assert loaded["events"][list(USAGE_FIELDS)].isna().all().all()
    assert loaded["orders"].set_index("order_id").loc["checkout", "discount_minor"] == 15


def test_v3_sql_failure_publishes_no_partial_dataset(coupon_dataset, tmp_path, monkeypatch):
    tables, metadata = coupon_dataset
    monkeypatch.setattr(raw, "_SCHEMA", raw._SCHEMA + """
CREATE TRIGGER fail_usage BEFORE INSERT ON events WHEN NEW.event_type='coupon_usage'
BEGIN SELECT RAISE(ABORT,'v3 injected failure'); END;
""")
    with pytest.raises(sqlite3.IntegrityError, match="v3 injected"):
        save_raw_dataset(tmp_path / "raw.sqlite", tables, metadata)
    assert list(tmp_path.iterdir()) == []
