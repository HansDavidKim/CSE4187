"""Eight event types share accounting without inventing visits for asynchronous contacts."""
import copy
import sqlite3

import pandas as pd
import pytest

from test_v4_raw_store import dataset
from test_v4_raw_store_coupon_usage import coupon_dataset
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, load_raw_dataset, save_raw_dataset, validate_raw_dataset


@pytest.fixture
def asynchronous(coupon_dataset):
    tables, metadata = copy.deepcopy(coupon_dataset)
    metadata.update(raw_schema_version=4, observation_start_at="2023-12-31T00:00:00Z",
                    observation_end_at="2024-01-05T00:00:00Z")
    rows = []
    for identifier, kind, at, fields in (
        (10, "search", "2024-01-02T10:06:00Z", {"session_id": "s2"}),
        (11, "review", "2024-01-03T12:00:00Z", {"related_purchase_event_id": 7,
             "rating": 4, "rating_kind": "product_star_scenario"}),
        (12, "cs_contact", "2024-01-04T09:00:00Z", {"contact_id": "ticket-1"}),
    ):
        row = {key: None for key in RAW_COLUMNS["events"]}
        row.update(event_id=identifier, customer_id="u1", event_type=kind, event_at=pd.Timestamp(at), **fields)
        rows.append(row)
    tables["events"] = pd.DataFrame([*tables["events"].to_dict("records"), *rows], columns=RAW_COLUMNS["events"])
    return tables, metadata


def test_eight_types_roundtrip_and_null_session_contacts_preserve_cart(asynchronous, tmp_path):
    tables, metadata = asynchronous
    validate_raw_dataset(tables, metadata)
    assert tables["events"].event_type.nunique() == 8
    assert len(tables["sessions"]) == 2
    assert cart_at(tables, "u1", metadata["observation_end_at"]) == {"B": 1}
    path = tmp_path / "raw.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, info = load_raw_dataset(path)
    assert info["raw_schema_version"] == 4
    async_events = loaded["events"].loc[loaded["events"].event_type.isin(["review", "cs_contact"])]
    assert async_events.session_id.isna().all()
    assert len(loaded["sessions"]) == 2
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert len(db.execute("PRAGMA table_info(events)").fetchall()) == 17
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        assert db.execute("SELECT COUNT(*) FROM events WHERE related_purchase_event_id=7").fetchone()[0] == 2


@pytest.mark.parametrize("case", ["fake_session", "before_purchase", "wrong_customer", "missing_parent",
                                 "rating", "missing_kind", "review_coupon", "duplicate_review", "duplicate_contact",
                                 "search_no_session", "contact_parent", "outside_observation", "invented_product"])
def test_async_constraints_reject_bad_links_clocks_and_visit_fabrication(asynchronous, case):
    tables, metadata = asynchronous
    events = tables["events"]
    review, contact, search = (events.event_type.eq(kind) for kind in ("review", "cs_contact", "search"))
    if case == "fake_session": events.loc[review, "session_id"] = "s2"
    elif case == "before_purchase": events.loc[review, "event_at"] = pd.Timestamp("2024-01-02T10:04:00Z")
    elif case == "wrong_customer": events.loc[review, "customer_id"] = "u2"
    elif case == "missing_parent": events.loc[review, "related_purchase_event_id"] = None
    elif case == "rating": events.loc[review, "rating"] = 6
    elif case == "missing_kind": events.loc[review, "rating_kind"] = None
    elif case == "review_coupon": events.loc[review, "coupon_id"] = "c1"
    elif case in {"duplicate_review", "duplicate_contact"}:
        extra = events.loc[review if case == "duplicate_review" else contact].iloc[0].copy()
        extra.event_id = 13
        if case == "duplicate_review": extra.event_at = pd.Timestamp("2024-01-04T10:00:00Z")
        tables["events"] = pd.DataFrame([*events.to_dict("records"), extra.to_dict()], columns=events.columns)
    elif case == "search_no_session": events.loc[search, "session_id"] = None
    elif case == "contact_parent": events.loc[contact, "related_purchase_event_id"] = 7
    elif case == "outside_observation": events.loc[contact, "event_at"] = pd.Timestamp(metadata["observation_end_at"])
    elif case == "invented_product": events.loc[contact, ["product_id", "unit_price_minor"]] = ["A", 100]
    with pytest.raises(ValueError):
        validate_raw_dataset(tables, metadata)


@pytest.mark.parametrize("version", [1, 2, 3])
def test_legacy_schema_column_counts_stay_unchanged(coupon_dataset, tmp_path, version):
    tables, metadata = copy.deepcopy(coupon_dataset)
    if version < 3:
        tables["events"] = tables["events"].loc[tables["events"].event_type.ne("coupon_usage")].copy()
        tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["discount_minor", "paid_minor"]] = [15, 136]
    metadata["raw_schema_version"] = version
    path = tmp_path / f"v{version}.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, _ = load_raw_dataset(path)
    assert len(loaded["events"].columns) == 17
    with sqlite3.connect(path) as db:
        assert len(db.execute("PRAGMA table_info(events)").fetchall()) == (14 if version == 3 else 10)


@pytest.fixture
def product_reviews(asynchronous):
    tables, metadata = copy.deepcopy(asynchronous)
    metadata["raw_schema_version"] = 5
    selected = tables["events"].event_type.eq("review")
    tables["events"].loc[selected, ["product_id", "order_line_no"]] = ["A", 1]
    second = tables["events"].loc[selected].iloc[0].to_dict()
    second.update(event_id=13, event_at=pd.Timestamp("2024-01-04T10:00:00Z"), product_id="B", order_line_no=2, rating=2)
    tables["events"] = pd.DataFrame([*tables["events"].to_dict("records"), second], columns=RAW_COLUMNS["events"])
    return tables, metadata


def test_raw5_distinct_purchased_products_allow_two_async_reviews(product_reviews, tmp_path):
    tables, metadata = product_reviews
    original = copy.deepcopy(tables)
    validate_raw_dataset(tables, metadata)
    path = tmp_path / "raw5.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, manifest = load_raw_dataset(path)
    assert manifest["raw_schema_version"] == 5
    reviews = loaded["events"].loc[loaded["events"].event_type.eq("review")]
    assert reviews.product_id.tolist() == ["A", "B"] and reviews.order_line_no.tolist() == [1, 2]
    assert reviews.related_purchase_event_id.eq(7).all()
    assert reviews[["session_id", "quantity", "unit_price_minor", "offer_product_id", "offer_unit_price_minor", "coupon_id", "discount_minor"]].isna().all().all()
    assert cart_at(loaded, "u1", metadata["observation_end_at"]) == {"B": 1}
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original[name])
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 5
        assert len(db.execute("PRAGMA table_info(events)").fetchall()) == 17
        assert len(db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()) == 7
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
        names = {row[1] for row in db.execute("PRAGMA index_list(events)")}
        assert "review_purchase_product" in names and "review_purchase" not in names
        assert [row[2] for row in db.execute("PRAGMA index_info(review_purchase_product)")] == ["related_purchase_event_id", "product_id"]
        assert db.execute("SELECT COUNT(*) FROM events WHERE related_purchase_event_id=7").fetchone()[0] == 3


def test_raw5_quantity_three_still_permits_only_one_review_for_the_sku(product_reviews):
    tables, metadata = product_reviews
    line = tables["order_items"].order_id.eq("checkout") & tables["order_items"].product_id.eq("A")
    tables["order_items"].loc[line, "quantity"] = 3
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["checkout_source", "gross_minor", "paid_minor"]] = ["mixed", 351, 341]
    validate_raw_dataset(tables, metadata)
    reviews = tables["events"].loc[tables["events"].event_type.eq("review")]
    assert reviews.product_id.eq("A").sum() == 1 and reviews.quantity.isna().all()
    extra = reviews.loc[reviews.product_id.eq("A")].iloc[0].to_dict()
    extra.update(event_id=14, event_at=pd.Timestamp("2024-01-04T11:00:00Z"))
    tables["events"] = pd.DataFrame([*tables["events"].to_dict("records"), extra], columns=RAW_COLUMNS["events"])
    with pytest.raises(ValueError, match="unique earlier purchase/product"):
        validate_raw_dataset(tables, metadata)


def test_raw5_same_sku_can_be_reviewed_again_after_another_purchase(product_reviews):
    tables, metadata = product_reviews
    extra = tables["events"].loc[tables["events"].event_type.eq("review") & tables["events"].product_id.eq("A")].iloc[0].to_dict()
    extra.update(event_id=14, event_at=pd.Timestamp("2024-01-04T11:00:00Z"), related_purchase_event_id=6)
    tables["events"] = pd.DataFrame([*tables["events"].to_dict("records"), extra], columns=RAW_COLUMNS["events"])
    validate_raw_dataset(tables, metadata)
    reviews = tables["events"].loc[tables["events"].event_type.eq("review")]
    assert len(reviews) == 3 and reviews.product_id.eq("A").sum() == 2


def test_raw5_duplicate_sku_lines_are_one_review_opportunity(product_reviews):
    tables, metadata = product_reviews
    tables["order_items"] = pd.concat([tables["order_items"], pd.DataFrame([
        dict(order_id="checkout", line_no=3, product_id="A", quantity=1, unit_price_minor=100)])], ignore_index=True)
    tables["orders"].loc[tables["orders"].order_id.eq("checkout"), ["checkout_source", "gross_minor", "paid_minor"]] = ["mixed", 251, 241]
    selected = tables["events"].event_type.eq("review") & tables["events"].product_id.eq("A")
    tables["events"].loc[selected, "order_line_no"] = 3
    validate_raw_dataset(tables, metadata)  # Any matching purchased line may carry the one SKU review.
    extra = tables["events"].loc[selected].iloc[0].to_dict()
    extra.update(event_id=14, event_at=pd.Timestamp("2024-01-04T11:00:00Z"), order_line_no=1)
    tables["events"] = pd.DataFrame([*tables["events"].to_dict("records"), extra], columns=RAW_COLUMNS["events"])
    with pytest.raises(ValueError, match="unique earlier purchase/product"):
        validate_raw_dataset(tables, metadata)


@pytest.mark.parametrize("case", ["missing_product", "missing_line", "zero_line", "wrong_line", "unbought_catalog_sku",
                                 "unbought_product_in_earlier_order", "quantity", "price", "offer", "coupon", "discount",
                                 "session", "customer", "before_purchase"])
def test_raw5_rejects_unbought_products_wrong_lines_and_nonnull_review_money(product_reviews, case):
    tables, metadata = product_reviews
    selected = tables["events"].event_type.eq("review") & tables["events"].product_id.eq("B")
    if case == "missing_product": tables["events"].loc[selected, "product_id"] = None
    elif case == "missing_line": tables["events"].loc[selected, "order_line_no"] = None
    elif case == "zero_line": tables["events"].loc[selected, "order_line_no"] = 0
    elif case == "wrong_line": tables["events"].loc[selected, "order_line_no"] = 1
    elif case == "unbought_catalog_sku":
        tables["products"] = pd.concat([tables["products"], pd.DataFrame([
            dict(product_id="C", category_id="c", unit_price_minor=70)])], ignore_index=True)
        tables["events"].loc[selected, "product_id"] = "C"
    elif case == "unbought_product_in_earlier_order":
        tables["events"].loc[selected, ["related_purchase_event_id", "order_line_no"]] = [6, 1]
    elif case == "quantity": tables["events"].loc[selected, "quantity"] = 1
    elif case == "price": tables["events"].loc[selected, "unit_price_minor"] = 51
    elif case == "offer": tables["events"].loc[selected, ["offer_product_id", "offer_unit_price_minor"]] = ["B", 51]
    elif case == "coupon": tables["events"].loc[selected, "coupon_id"] = "c1"
    elif case == "discount": tables["events"].loc[selected, "discount_minor"] = 0
    elif case == "session": tables["events"].loc[selected, "session_id"] = "s2"
    elif case == "customer": tables["events"].loc[selected, "customer_id"] = "u2"
    else: tables["events"].loc[selected, "event_at"] = pd.Timestamp("2024-01-02T10:04:00Z")
    with pytest.raises(ValueError):
        validate_raw_dataset(tables, metadata)


@pytest.mark.parametrize("column,value", [("quantity", 1), ("unit_price_minor", 51), ("offer_product_id", "A"),
                                         ("product_id", None), ("order_line_no", None), ("coupon_id", "c1"), ("discount_minor", 0)])
def test_raw5_sql_checks_reject_missing_product_line_and_nonnull_review_money(product_reviews, tmp_path, column, value):
    tables, metadata = product_reviews
    path = tmp_path / "raw5.sqlite"
    save_raw_dataset(path, tables, metadata)
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(f'UPDATE events SET "{column}"=? WHERE event_id=13', (value,))


def test_raw5_sql_partial_unique_index_rejects_duplicate_purchase_sku(product_reviews, tmp_path):
    tables, metadata = product_reviews
    path = tmp_path / "raw5.sqlite"
    save_raw_dataset(path, tables, metadata)
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            db.execute("UPDATE events SET product_id='A',order_line_no=1 WHERE event_id=13")


def test_raw4_order_only_review_roundtrip_is_preserved(asynchronous, product_reviews, tmp_path):
    tables, metadata = asynchronous
    path = tmp_path / "old-raw4.sqlite"
    save_raw_dataset(path, tables, metadata)
    loaded, info = load_raw_dataset(path)
    review = loaded["events"].loc[loaded["events"].event_type.eq("review")]
    assert info["raw_schema_version"] == 4 and len(review) == 1
    assert review.product_id.isna().all() and review.order_line_no.isna().all()
    with sqlite3.connect(path) as db:
        assert "review_purchase" in {row[1] for row in db.execute("PRAGMA index_list(events)")}
    new_tables, new_metadata = product_reviews
    new_metadata["raw_schema_version"] = 4
    with pytest.raises(ValueError):
        validate_raw_dataset(new_tables, new_metadata)
