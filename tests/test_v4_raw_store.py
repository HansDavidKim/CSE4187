import json
import sqlite3

import pandas as pd
import pytest

import simulate.v4.simulator.raw_store as raw
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, discount_minor, load_raw_dataset, save_raw_dataset, validate_raw_dataset


@pytest.fixture
def dataset():
    def event(event_id, session_id, when, kind, product=None, quantity=None, offer=None):
        prices = {"A": 100, "B": 51}
        return dict(event_id=event_id, customer_id="u1", session_id=session_id, event_at=pd.Timestamp(when, tz="UTC"),
                    event_type=kind, product_id=product, quantity=quantity, unit_price_minor=prices.get(product),
                    offer_product_id=offer, offer_unit_price_minor=prices.get(offer))
    records = {
        "customers": [dict(customer_id=uid, signed_up_at=pd.Timestamp("2023-12-31", tz="UTC")) for uid in ("u1", "u2")],
        "products": [dict(product_id="A", category_id="c1", unit_price_minor=100),
                     dict(product_id="B", category_id=None, unit_price_minor=51)],
        "sessions": [dict(session_id=f"s{day}", customer_id="u1", started_at=pd.Timestamp(f"2024-01-0{day} 10:00", tz="UTC"),
                          ended_at=pd.Timestamp(f"2024-01-0{day} 10:30", tz="UTC")) for day in (1, 2)],
        "events": [event(1, "s1", "2024-01-01 10:00", "view", "A"),
                   event(2, "s1", "2024-01-01 10:01", "cart", "A", 1),
                   event(3, "s1", "2024-01-01 10:01", "cart", "A", 1),
                   event(4, "s1", "2024-01-01 10:02", "cart", "B", 1),
                   event(5, "s1", "2024-01-01 10:03", "remove_from_cart", "A", 1),
                   event(6, "s2", "2024-01-02 10:00", "purchase", offer="A"),
                   event(7, "s2", "2024-01-02 10:05", "purchase", offer="B"),
                   event(8, "s2", "2024-01-02 10:06", "cart", "B", 1)],
        "orders": [dict(order_id="buy", purchase_event_id=6, checkout_source="buy_now", coupon_id=None,
                        gross_minor=100, discount_minor=0, paid_minor=100),
                   dict(order_id="checkout", purchase_event_id=7, checkout_source="cart", coupon_id="c1",
                        gross_minor=151, discount_minor=15, paid_minor=136)],
        "order_items": [dict(order_id=order, line_no=line, product_id=product, quantity=1, unit_price_minor=price)
                        for order, line, product, price in [("buy", 1, "A", 100), ("checkout", 1, "A", 100),
                                                            ("checkout", 2, "B", 51)]],
        "coupons": [dict(coupon_id="c1", customer_id="u1", sent_at=pd.Timestamp("2024-01-01 09:00", tz="UTC"),
                         expires_at=pd.Timestamp("2024-01-03", tz="UTC"), discount_rate=0.1, policy_id="ten-percent",
                         send_cost_minor=1)],
    }
    tables = {name: pd.DataFrame(records[name], columns=columns) for name, columns in RAW_COLUMNS.items()}
    metadata = dict(dataset_id="unit-test", seed=7, currency="SOURCE_UNSPECIFIED", amount_unit="0.01", schema_version=9)
    return tables, metadata


def test_roundtrip_has_seven_relations_and_preserves_units_and_times(dataset, tmp_path):
    tables, metadata = dataset
    original = {name: frame.copy(deep=True) for name, frame in tables.items()}
    path = tmp_path / "raw.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, saved_metadata = load_raw_dataset(path)
    assert saved_metadata == {**metadata, "raw_schema_version": 1}
    assert metadata["schema_version"] == 9 and "raw_schema_version" not in metadata
    assert set(loaded) == set(RAW_COLUMNS)
    assert str(loaded["events"].event_at.dtype) == "datetime64[ns, UTC]"
    assert loaded["events"].event_id.dtype == pd.Int64Dtype()
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original[name])
        assert len(loaded[name]) == len(tables[name])
    with sqlite3.connect(path) as connection:
        assert {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")} == set(RAW_COLUMNS)
        assert connection.execute("SELECT typeof(gross_minor) FROM orders").fetchone()[0] == "integer"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert cart_at(loaded, "u1", "2024-01-02T10:07:00Z") == {"B": 1}


def test_cart_survives_sessions_buy_now_and_same_time_order(dataset):
    tables, metadata = dataset
    validate_raw_dataset(tables, metadata)
    assert cart_at(tables, "u1", "2024-01-01T10:01:00Z") == {}
    assert cart_at(tables, "u1", "2024-01-01T19:01:00+09:00", before_event_id=3) == {"A": 1}
    assert cart_at(tables, "u1", "2024-01-02T10:00:00Z", before_event_id=6) == {"A": 1, "B": 1}
    assert cart_at(tables, "u1", "2024-01-02T10:05:00Z") == {"A": 1, "B": 1}
    assert cart_at(tables, "u1", "2024-01-02T10:05:00Z", before_event_id=8) == {}
    assert cart_at(tables, "u2", "2024-01-02T10:00:00Z") == {}


def test_future_records_do_not_change_past_cart(dataset):
    tables, metadata = dataset
    cutoff = "2024-01-01T10:04:00Z"
    expected = cart_at(tables, "u1", cutoff)
    future = tables["coupons"].iloc[0].copy()
    future["coupon_id"] = "future"
    future["sent_at"] = pd.Timestamp("2024-01-04", tz="UTC")
    future["expires_at"] = pd.Timestamp("2024-01-05", tz="UTC")
    tables["coupons"] = pd.concat([tables["coupons"], future.to_frame().T], ignore_index=True)
    tables["sessions"].loc[1, "ended_at"] = pd.Timestamp("2024-01-02 12:00", tz="UTC")
    validate_raw_dataset(tables, metadata)
    assert cart_at(tables, "u1", cutoff) == expected


def test_discount_is_half_up_and_validates_minor_units():
    assert discount_minor(5, 0.1) == 1
    assert discount_minor(15, "0.1") == 2
    assert discount_minor(151, 0.1) == 15
    assert discount_minor(10, 1) == 10
    for gross, rate in [(1.5, 0.1), (True, 0.1), (-1, 0.1), (10, -0.1), (10, float("nan")), (10, True)]:
        with pytest.raises(ValueError):
            discount_minor(gross, rate)


@pytest.mark.parametrize("case,match", [
    ("owner", "session owner"), ("time", "session owner or time"), ("naive", "UTC-aware"),
    ("fractional", "integer"), ("duplicate", "primary key"), ("missing_order", "exactly one order"),
    ("item", "order item"), ("accounting", "reconcile"), ("coupon_owner", "owned, valid"),
    ("coupon_expiry", "owned, valid"), ("coupon_twice", "owned, valid"),
    ("negative_cart", "becomes negative"), ("partial_checkout", "complete current cart"),
    ("display_price", "fixed display price"), ("cart_quantity", "cart adds one"),
])
def test_invalid_records_fail_before_writing(dataset, tmp_path, case, match):
    tables, metadata = dataset
    if case == "owner": tables["events"].loc[0, "customer_id"] = "u2"
    elif case == "time": tables["events"].loc[0, "event_at"] = pd.Timestamp("2024-01-01 09:59", tz="UTC")
    elif case == "naive": tables["customers"]["signed_up_at"] = pd.to_datetime(["2023-12-31", "2023-12-31"])
    elif case == "fractional": tables["products"]["unit_price_minor"] = [100.5, 51]
    elif case == "duplicate": tables["customers"] = pd.concat([tables["customers"], tables["customers"].iloc[:1]])
    elif case == "missing_order":
        tables["orders"] = tables["orders"].iloc[1:]
        tables["order_items"] = tables["order_items"].loc[tables["order_items"].order_id.ne("buy")]
    elif case == "item": tables["order_items"].loc[0, "quantity"] = 0
    elif case == "accounting": tables["orders"].loc[0, "gross_minor"] = 99
    elif case == "coupon_owner": tables["coupons"].loc[0, "customer_id"] = "u2"
    elif case == "coupon_expiry": tables["coupons"].loc[0, "expires_at"] = pd.Timestamp("2024-01-02 10:05", tz="UTC")
    elif case == "coupon_twice":
        tables["orders"].loc[0, ["coupon_id", "discount_minor", "paid_minor"]] = ["c1", 10, 90]
    elif case == "negative_cart":
        tables["events"].loc[0, ["event_type", "quantity"]] = ["remove_from_cart", 1]
    elif case == "partial_checkout":
        tables["order_items"] = tables["order_items"].iloc[:2]
        tables["orders"].loc[1, ["gross_minor", "discount_minor", "paid_minor"]] = [100, 10, 90]
    elif case == "display_price": tables["events"].loc[0, "unit_price_minor"] = 99
    elif case == "cart_quantity": tables["events"].loc[1, "quantity"] = 2
    path = tmp_path / "raw.sqlite"
    with pytest.raises(ValueError, match=match):
        save_raw_dataset(path, tables, metadata)
    assert not path.exists()
    assert not path.with_suffix(".manifest.json").exists()


def test_sql_failure_rolls_back_without_publishing_partial_purchase(dataset, tmp_path, monkeypatch):
    tables, metadata = dataset
    monkeypatch.setattr(raw, "_SCHEMA", raw._SCHEMA + """
CREATE TRIGGER reject_second_item BEFORE INSERT ON order_items WHEN NEW.line_no=2
BEGIN SELECT RAISE(ABORT,'injected item failure'); END;
""")
    path = tmp_path / "raw.sqlite"
    with pytest.raises(sqlite3.IntegrityError, match="injected item failure"):
        save_raw_dataset(path, tables, metadata)
    assert list(tmp_path.iterdir()) == []


def test_existing_files_and_racing_manifest_are_preserved(dataset, tmp_path, monkeypatch):
    tables, metadata = dataset
    path = tmp_path / "raw.sqlite"
    path.write_bytes(b"existing database")
    with pytest.raises(FileExistsError):
        save_raw_dataset(path, tables, metadata)
    assert path.read_bytes() == b"existing database"
    path.unlink()
    link = raw.os.link
    def competing_writer(source, destination):
        if destination == path.with_suffix(".manifest.json"):
            destination.write_text("another writer")
        return link(source, destination)
    monkeypatch.setattr(raw.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        save_raw_dataset(path, tables, metadata)
    assert not path.exists()
    assert path.with_suffix(".manifest.json").read_text() == "another writer"


@pytest.mark.parametrize("existing_name", ["raw.sqlite", "raw.manifest.json"])
def test_incomplete_file_pair_is_not_overwritten_or_loaded(dataset, tmp_path, existing_name):
    tables, metadata = dataset
    existing = tmp_path / existing_name
    existing.write_text("incomplete original")
    path = tmp_path / "raw.sqlite"
    with pytest.raises(FileExistsError):
        save_raw_dataset(path, tables, metadata)
    assert existing.read_text() == "incomplete original"
    with pytest.raises((FileNotFoundError, json.JSONDecodeError)):
        load_raw_dataset(path)
    assert {file.name for file in tmp_path.iterdir()} == {existing_name}


def test_checksum_and_version_guard(dataset, tmp_path):
    tables, metadata = dataset
    path = tmp_path / "raw.sqlite"
    save_raw_dataset(path, tables, metadata)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema version"):
        load_raw_dataset(path)
    manifest["raw_schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_raw_dataset(path)


def test_empty_relations_can_roundtrip(dataset, tmp_path):
    tables, metadata = dataset
    tables = {name: frame.iloc[:0] for name, frame in tables.items()}
    path = tmp_path / "empty.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, _ = load_raw_dataset(path)
    assert all(frame.empty for frame in loaded.values())
