"""Coupon accounting events use purchased units without another shopping action."""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from test_v4_checkout_simulator import checkout_fixture
from test_v4_tianchi_coupon_simulator import _config, _run
from simulate.v4.simulator.raw_store import RAW_COLUMNS, validate_raw_dataset


@pytest.mark.parametrize("a_price,target,line,gross", [(105, "B", 2, 1025), (250, "A", 1, 1750)])
def test_usage_links_purchase_and_selects_highest_unit_with_stable_tie(
        checkout_fixture, tmp_path, a_price, target, line, gross):
    config = _config(checkout_fixture, tmp_path / "fixture", first_action="purchase")
    model_path = Path(config["source_simulation"]["model_path"])
    model = json.loads(model_path.read_text())
    for product in model["catalog"]:
        if product["product_id"] == "A":
            product["unit_price_minor"] = a_price
    templates = [{"items": [{"product_id": "A", "quantity": 5}, {"product_id": "B", "quantity": 2}],
                  "observed_count": 1}]
    for profile in model["profiles"]:
        profile["checkout_templates"] = templates
    for group in model["home_groups"].values():
        group["checkout_templates"] = templates
    model["distributions"]["checkout_templates"] = templates
    model_path.write_text(json.dumps(model))
    result = _run(config)
    tables = result.tables
    usage = tables["events"].loc[tables["events"].event_type.eq("coupon_usage")]
    assert not usage.empty and usage.coupon_id.is_unique
    assert len(usage) == tables["orders"].coupon_id.notna().sum()
    assert usage.product_id.eq(target).all() and usage.order_line_no.eq(line).all()
    assert usage.quantity.eq(1).all() and usage.unit_price_minor.eq(250).all()
    assert usage.discount_minor.eq(25).all()  # One unit, even when the purchased quantity is two or five.
    assert tables["events"].loc[tables["events"].event_type.ne("coupon_usage"),
                                ["related_purchase_event_id", "coupon_id", "order_line_no", "discount_minor"]].isna().all().all()
    purchases = tables["events"].set_index("event_id")
    orders = tables["orders"].set_index("purchase_event_id")
    treated = set(tables["treatments"].loc[tables["treatments"].treatment.eq(1), "customer_id"])
    for event in usage.itertuples(index=False):
        purchase = purchases.loc[event.related_purchase_event_id]
        order = orders.loc[event.related_purchase_event_id]
        assert event.customer_id in treated
        assert purchase.event_type == "purchase" and event.event_id > event.related_purchase_event_id
        assert (event.customer_id, event.session_id, event.event_at) == (
            purchase.customer_id, purchase.session_id, purchase.event_at)
        assert order.coupon_id == event.coupon_id and order.gross_minor == gross
        assert order.discount_minor == 25 and order.paid_minor == gross - 25
    assert result.metadata["event_types"] == ["view", "cart", "remove_from_cart", "purchase", "coupon_usage", "review"]
    assert result.metadata["coupon_application"]["source"].startswith("explicit user-requested")
    validate_raw_dataset({name: tables[name] for name in RAW_COLUMNS}, result.metadata)
    output = tmp_path / "saved"
    result.save(output)
    with sqlite3.connect(output / "raw.sqlite") as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM events WHERE event_type='coupon_usage'").fetchone()[0] == len(usage)
        assert db.execute("SELECT COUNT(*) FROM events u JOIN orders o ON o.purchase_event_id=u.related_purchase_event_id "
                          "WHERE u.event_type='coupon_usage' AND u.discount_minor=o.discount_minor "
                          "AND u.coupon_id=o.coupon_id").fetchone()[0] == len(usage)
