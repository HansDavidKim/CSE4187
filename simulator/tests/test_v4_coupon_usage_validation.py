"""Independent raw3 coupon-use checks with hand-built business records."""
import copy

import pandas as pd
import pytest

from simulate.v4 import validation as audit
from test_v4_source_validation import records


@pytest.fixture
def coupon_usage_records():
    at = pd.Timestamp("2024-01-07T15:00:00Z")
    tables = {
        "customers": pd.DataFrame([dict(customer_id="u", signed_up_at=at-pd.Timedelta(days=30))]),
        "products": pd.DataFrame([dict(product_id=pid, category_id="category", unit_price_minor=price)
                                   for pid, price in [("A", 105), ("B", 250), ("C", 250)]]),
        "sessions": pd.DataFrame([dict(session_id="s", customer_id="u", started_at=at,
                                        ended_at=at+pd.Timedelta(seconds=10))]),
        "events": pd.DataFrame([
            dict(event_id=1, customer_id="u", session_id="s", event_at=at, event_type="cart", product_id="A", quantity=1, unit_price_minor=105),
            dict(event_id=2, customer_id="u", session_id="s", event_at=at+pd.Timedelta(seconds=1), event_type="purchase", product_id=None, quantity=None, unit_price_minor=None),
            dict(event_id=3, customer_id="u", session_id="s", event_at=at+pd.Timedelta(seconds=1), event_type="coupon_usage", product_id="B", quantity=1, unit_price_minor=250,
                 related_purchase_event_id=2, coupon_id="coupon", order_line_no=2, discount_minor=25),
        ]),
        "orders": pd.DataFrame([dict(order_id="o", purchase_event_id=2, checkout_source="mixed", coupon_id="coupon",
                                      gross_minor=1065, discount_minor=25, paid_minor=1040)]),
        "order_items": pd.DataFrame([dict(order_id="o", line_no=line, product_id=pid, quantity=quantity, unit_price_minor=price)
                                      for line, pid, quantity, price in [(1, "A", 3, 105), (2, "B", 2, 250), (3, "C", 1, 250)]]),
        "coupons": pd.DataFrame([dict(coupon_id="coupon", customer_id="u", sent_at=at, expires_at=at+pd.Timedelta(days=30),
                                       discount_rate=.1, policy_id="one_coupon_10pct_highest_unit_30d", send_cost_minor=1)]),
    }
    tables["events"]["offer_product_id"] = None
    tables["events"]["offer_unit_price_minor"] = None
    metadata = dict(raw_schema_version=3, as_of_at=at.isoformat(), observation_start_at=(at-pd.Timedelta(days=40)).isoformat(),
                    observation_end_at=(at+pd.Timedelta(days=60)).isoformat())
    return tables, metadata


def test_raw3_single_highest_price_unit_and_event_pair_are_independently_verified(coupon_usage_records):
    tables, metadata = coupon_usage_records
    original = copy.deepcopy(tables)
    result = audit.audit_coupon_usage(tables, metadata)
    assert result["passed"] and not any(result["mismatch_counts"].values())
    assert result["coupon_usage_events"] == result["orders_with_coupon"] == result["purchase_events"] == 1
    assert result["usage_discount_minor"] == result["order_discount_minor"] == 25
    assert 25 != 107  # Whole-order 10% of 1065 rounds to 107.
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], original[name])


@pytest.mark.parametrize("mutation,field", [
    ("quantity", "usage_quantity_not_one"), ("target_line", "usage_highest_unit_line"),
    ("tie_line", "usage_highest_unit_line"), ("unit_discount", "usage_unit_discount"),
    ("gross_discount", "order_unit_discount"), ("price", "usage_target_price"),
    ("product", "usage_target_product"), ("parent", "usage_purchase_fk"),
    ("parent_type", "usage_purchase_fk"), ("own_id", "usage_event_order"),
    ("customer", "usage_purchase_customer"), ("session", "usage_purchase_session"),
    ("time", "usage_purchase_time"), ("coupon", "usage_order_coupon"),
    ("expiry", "usage_coupon_clock"), ("coupon_customer", "usage_coupon_customer"),
    ("nonusage_link", "nonusage_has_coupon_linkage"), ("duplicate", "duplicate_usage_coupon"),
    ("missing", "usage_count_by_order"), ("diagnostic_parent", "order_not_linked_to_purchase"),
])
def test_coupon_usage_corruption_is_detected_without_trusting_raw_loader(coupon_usage_records, mutation, field):
    tables, metadata = coupon_usage_records
    events = tables["events"]
    if mutation == "quantity": events.loc[2, "quantity"] = 2
    elif mutation == "target_line": events.loc[2, "order_line_no"] = 1
    elif mutation == "tie_line":
        events.loc[2, ["order_line_no", "product_id"]] = [3, "C"]
    elif mutation == "unit_discount": events.loc[2, "discount_minor"] = 50  # Two B units would be 50.
    elif mutation == "gross_discount":
        tables["orders"].loc[0, ["discount_minor", "paid_minor"]] = [107, 958]
        events.loc[2, "discount_minor"] = 107
    elif mutation == "price": events.loc[2, "unit_price_minor"] = 251
    elif mutation == "product": events.loc[2, "product_id"] = "A"
    elif mutation == "parent": events.loc[2, "related_purchase_event_id"] = 999
    elif mutation == "parent_type": events.loc[2, "related_purchase_event_id"] = 3
    elif mutation == "own_id": events.loc[2, "event_id"] = 0
    elif mutation == "customer": events.loc[2, "customer_id"] = "other"
    elif mutation == "session": events.loc[2, "session_id"] = "other"
    elif mutation == "time": events.loc[2, "event_at"] += pd.Timedelta(seconds=1)
    elif mutation == "coupon": events.loc[2, "coupon_id"] = "other"
    elif mutation == "expiry": tables["coupons"].loc[0, "expires_at"] = events.loc[2, "event_at"]
    elif mutation == "coupon_customer": tables["coupons"].loc[0, "customer_id"] = "other"
    elif mutation == "nonusage_link": events.loc[0, "coupon_id"] = "coupon"
    elif mutation == "duplicate":
        duplicate = events.iloc[2].copy()
        duplicate["event_id"] = 4
        tables["events"] = pd.concat([events, duplicate.to_frame().T], ignore_index=True)
    elif mutation == "missing": tables["events"] = events.iloc[:2].copy()
    elif mutation == "diagnostic_parent": tables["orders"].loc[0, "purchase_event_id"] = 3
    result = audit.audit_coupon_usage(tables, metadata)
    assert not result["passed"] and result["mismatch_counts"][field] > 0


def test_usage_is_required_even_when_one_unit_discount_rounds_to_zero(coupon_usage_records):
    tables, metadata = coupon_usage_records
    tables["products"]["unit_price_minor"] = 1
    tables["order_items"]["unit_price_minor"] = 1
    tables["orders"].loc[0, ["gross_minor", "discount_minor", "paid_minor"]] = [6, 0, 6]
    tables["events"].loc[0, "unit_price_minor"] = 1
    tables["events"].loc[2, ["product_id", "unit_price_minor", "order_line_no", "discount_minor"]] = ["A", 1, 1, 0]
    assert audit.audit_coupon_usage(tables, metadata)["passed"]
    tables["events"] = tables["events"].iloc[:2].copy()
    assert not audit.audit_coupon_usage(tables, metadata)["passed"]


def test_valid_usage_can_belong_to_an_observed_open_session(coupon_usage_records):
    tables, metadata = coupon_usage_records
    tables["sessions"]["ended_at"] = pd.NaT
    assert audit.audit_coupon_usage(tables, metadata)["passed"]


@pytest.mark.parametrize("version", [1, 2])
def test_archived_schema_keeps_whole_order_contract_and_rejects_usage_rows(coupon_usage_records, version):
    tables, metadata = coupon_usage_records
    metadata["raw_schema_version"] = version
    assert not audit.audit_coupon_usage(tables, metadata)["passed"]
    tables["events"] = tables["events"].iloc[:2].copy()
    tables["orders"].loc[0, ["discount_minor", "paid_minor"]] = [107, 958]
    result = audit.audit_coupon_usage(tables, metadata)
    assert result["passed"] and result["status"] == "legacy_whole_order_discount"


def test_coupon_usage_cannot_double_shopping_item_counts_or_distribution_metrics(coupon_usage_records):
    tables, metadata = coupon_usage_records
    interactions = audit._generated_interactions(tables)
    without_usage = dict(tables)
    without_usage["events"] = tables["events"].loc[tables["events"].event_type.ne("coupon_usage")]
    pd.testing.assert_frame_equal(interactions, audit._generated_interactions(without_usage))
    assert set(interactions.event_type) == {"cart", "purchase"}
    assert interactions.quantity.sum() == 7  # One cart + six purchased units, no usage-unit addition.
    lower, upper = pd.Timestamp(metadata["observation_start_at"]), pd.Timestamp(metadata["observation_end_at"])
    _, _, summary = audit._metric_arrays(interactions, tables["sessions"], lower, upper)
    assert summary["item_proxy_weight"] == 7
    assert summary["item_proxy_action_share"] == {"cart": 1 / 7, "purchase": 6 / 7}
    usage_as_interaction = interactions.iloc[:1].copy()
    usage_as_interaction["event_type"] = "coupon_usage"
    _, _, duplicate_summary = audit._metric_arrays(pd.concat([interactions, usage_as_interaction]), tables["sessions"], lower, upper)
    assert duplicate_summary == summary


def test_raw3_sidecars_require_the_explicit_highest_unit_policy_ids(records):
    tables, metadata, labels, treatments = records
    metadata["raw_schema_version"] = 3
    metadata["policy_id"] = "coupon_once_highest_unit_30d"
    tables["coupons"]["policy_id"] = "one_coupon_10pct_highest_unit_30d"
    labels["policy_id"] = ["one_coupon_10pct_highest_unit_30d", "no_new_coupon"]
    treatments["experiment_id"] = "coupon_once_highest_unit_30d"
    purchase = tables["events"].loc[tables["events"].event_type.eq("purchase")].iloc[0]
    usage = dict(event_id=5, customer_id=purchase.customer_id, session_id=purchase.session_id,
                 event_at=purchase.event_at, event_type="coupon_usage", product_id="A", quantity=1,
                 unit_price_minor=5, offer_product_id=None, offer_unit_price_minor=None,
                 related_purchase_event_id=purchase.event_id, coupon_id="coupon1", order_line_no=1, discount_minor=1)
    tables["events"] = pd.concat([tables["events"], pd.DataFrame([usage])], ignore_index=True)
    result = audit.audit_sidecars(tables, metadata, labels, treatments)
    assert result["passed"] and result["coupon_usage"]["coupon_usage_events"] == 1
    labels.loc[0, "policy_id"] = "one_coupon_10pct_30d"
    result = audit.audit_sidecars(tables, metadata, labels, treatments)
    assert not result["passed"] and result["mismatch_counts"]["policy_id"] == 1
