"""Checkout integration uses fitted synthetic observations, never production artifacts."""
from collections import Counter
import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from simulate.v4.irl.fit_shopping import ACTIONS, fit_shopping
from simulate.v4.irl.fit_return import FEATURE_NAMES as RETURN_FEATURE_NAMES
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS, cart_at, discount_minor, load_raw_dataset, validate_raw_dataset


@pytest.fixture(scope="module")
def checkout_fixture(tmp_path_factory):
    rows = []
    sequence = [("view", "A"), ("cart", "A"), ("cart", "A"), ("remove_from_cart", "A"),
                ("view", "B"), ("cart", "B"), ("purchase", "B"), ("purchase", "A"), ("view", "A")]
    for user in range(40):
        for session, day in enumerate(("2020-01-01", "2020-01-03", "2020-02-15")):
            for step, (action, product) in enumerate(sequence):
                at = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=user % 20, seconds=(6 if step == 7 else step) * 5)
                rows.append(dict(event_time=at.isoformat(), event_type=action, product_id=product,
                                 category_id=f"category-{product}", category_code="", brand="",
                                 price="1.05" if product == "A" else "2.50", user_id=str(user),
                                 user_session=f"{user}:{session}"))
    model = fit_shopping(pd.DataFrame(rows), {"dataset": "synthetic checkout wiring fixture"}, checkout_episodes=True)
    folder = tmp_path_factory.mktemp("checkout-simulator")
    def write(name, value):
        path = folder / name
        path.write_text(json.dumps(value, allow_nan=False))
        return str(path)
    features = len(RETURN_FEATURE_NAMES)
    return_model = {"schema_version": 1, "feature_names": RETURN_FEATURE_NAMES,
                    "scaler_mean": [0.] * features, "scaler_scale": [1.] * features,
                    "feature_min": [0.] * features, "feature_max": [100.] * features,
                    "coef": [0.] * features, "intercept": math.log(4),
                    "cold_start_probability": .8, "calibration_reference_probabilities": [.8]}
    config = {"simulation": {"seed": 17, "start_at": "2024-01-01T00:00:00+00:00"},
              "churn": {"target_rate": .2},
              "source_simulation": {"model_path": write("shopping.json", model),
                                    "return_model_path": write("return.json", return_model),
                                    "max_events_per_session": 64, "outcome_days": 28,
                                    "coupon_discount": .1, "coupon_send_cost_minor": 100},
              "toy": {"message_model_path": write("message.json", {
                  "response_lift": {"positive": .1, "neutral": 0., "negative": -.1},
                  "visit_lift_scale": 1., "observed": {"pooled": {"visit_logit_shift": 0.}}}),
                      "coupon_model_path": write("coupon.json", {
                          "groups": {"baseline": {"coupon_use_probability": 1.}}})}}
    return config, model


def _customers(count=8):
    # Every supplied signup lies in the eligible history window; no rejected donor traces.
    return pd.DataFrame({"customer_id": [f"fixture-{i:02}" for i in range(count)],
                         "signed_up_at": pd.Timestamp("2024-01-10", tz="UTC")})


def _basket(items):
    result = Counter()
    for item in items.itertuples(index=False):
        result[item.product_id] += int(item.quantity)
    return result


@pytest.fixture
def cart_fulfillment():
    from simulate.v4.irl.fit_checkout_spending import fit_cart_fulfillment
    import hashlib
    uid = next(str(i) for i in range(100) if hashlib.sha256(f"42:{i}".encode()).digest()[0] < 205)
    actions = [("cart", "cheap")] * 4 + [("cart", "dear")] * 2
    rows = [dict(user_id=uid, at=pd.Timestamp("2020-01-01T00:00:00Z") + pd.Timedelta(seconds=i),
                 action=action, product_id=product, session="observed", source_event_id=f"{i:03}")
            for i, (action, product) in enumerate(actions)]
    rows += [dict(user_id=uid, at=pd.Timestamp("2020-01-01T00:00:10Z"), action="purchase",
                  product_id=product, session="observed", source_event_id=f"{i+6:03}")
             for i, product in enumerate(["cheap"] * 4 + ["dear"] * 2)]
    fitted = fit_cart_fulfillment(pd.DataFrame(rows), "2020-02-01T00:00:00Z")
    assert fitted["fraction_counts"] == [[1, 1, 1]]
    return fitted


def test_checkout_episode_consumes_matching_held_sku_and_replays_remaining_cart(checkout_fixture, tmp_path, monkeypatch):
    from simulate.v4.irl import fit_shopping as shopping
    from simulate.v4.features import build_features
    from uplift.common.features import build_dataset
    from uplift.common.raw import load_raw
    config, model = copy.deepcopy(checkout_fixture)
    model["catalog"].append({"product_id": "template", "category_id": "category-A", "unit_price_minor": 109})
    model.pop("checkout_kernel", None)
    template = [{"items": [{"product_id": "template", "quantity": 1}], "observed_count": 1}]
    for profile in model["profiles"]:
        profile.update(product_weights={"A": 1.}, session_first_action_counts={"view": 1},
                       session_choice_lengths=[5], choice_gap_seconds=[1.], checkout_templates=template)
    for group in model["home_groups"].values():
        group["checkout_templates"] = template
    model["distributions"]["checkout_templates"] = template
    path = tmp_path / "shopping.json"
    path.write_text(json.dumps(model))
    config["source_simulation"]["model_path"] = str(path)
    config["source_simulation"]["eligibility_window_days"] = 28
    def policy(model, context):
        action = ("view", "cart", "cart", "purchase", "remove")[context["session_decision_count"]]
        return {key: float(key == action) for key in (*ACTIONS, "end")}
    monkeypatch.setattr(shopping, "choice_probabilities", policy)
    result = EcommerceSimulator(config).generate(n_customers=8, months=6, seed=17)
    raw = {name: result.tables[name] for name in RAW_COLUMNS}
    validate_raw_dataset(raw, result.metadata)
    assert len(raw["orders"]) > 0 and raw["orders"].checkout_source.eq("cart").all()
    assert raw["order_items"].product_id.eq("A").all()
    orders = raw["orders"].merge(raw["events"][["event_id", "customer_id", "event_at"]],
                               left_on="purchase_event_id", right_on="event_id")
    for order in orders.itertuples(index=False):
        assert cart_at(raw, order.customer_id, order.event_at, before_event_id=order.event_id) == {"A": 2}
        assert cart_at(raw, order.customer_id, order.event_at, before_event_id=order.event_id + 1) == {"A": 1}
    assert raw["events"].event_type.eq("remove_from_cart").sum() >= len(orders)
    cutoff = pd.Timestamp(result.metadata["observation_end_at"])
    assert all(cart_at(raw, uid, cutoff) == {} for uid in raw["customers"].customer_id)
    output = tmp_path / "published"
    result.save(output)
    loaded, metadata = load_raw_dataset(output / "raw.sqlite")
    assert loaded["order_items"].product_id.eq("A").all()
    kit = build_features({**loaded, "_metadata": metadata}, as_of_at=pd.Timestamp(metadata["as_of_at"]))
    uplift = build_dataset(load_raw(output)).set_index("customer_id")
    for feature in ("cart_qty_current", "cart_sku_current", "cart_gross_value_current"):
        pd.testing.assert_series_equal(kit.set_index("customer_id")[feature].astype(float).sort_index(),
                                       uplift[feature].sort_index(), check_names=False)


def test_population_checkout_quota_cannot_spend_an_unaffordable_cart(tmp_path, monkeypatch, cart_fulfillment):
    from simulate.v4.configuration import POPULATION_CONFIG, load_config
    from simulate.v4.irl import fit_latent_shopping as shopping
    from simulate.v4.simulator import population_profiles
    config = load_config(POPULATION_CONFIG)
    model = json.loads(Path(config["source_simulation"]["model_path"]).read_text())
    model["population"]["checkout_spending"]["cart_fulfillment"] = cart_fulfillment
    path = tmp_path / "population.json"
    path.write_text(json.dumps(model))
    config["source_simulation"]["model_path"] = str(path)
    config["source_simulation"].pop("reward_gmm_model_path", None)
    config["source_simulation"]["reward_gmm_return_only"] = False
    prepare = population_profiles.prepare_population
    template = [{"items": [{"product_id": "target", "quantity": 1}], "observed_count": 1}]
    def prepared(population, products, rng):
        groups, draw = prepare(population, products, rng)
        for group in groups.values():
            group["checkout_templates"] = template
        def profile(index):
            result = draw(index)
            result.update(product_weights={"cheap": 1.}, checkout_templates=template, checkout_preference_share=0.)
            return result
        return groups, profile
    monkeypatch.setattr(population_profiles, "prepare_population", prepared)
    def policy(model, context, *args):
        action = ("view", "cart", "purchase", "end")[min(context["session_decision_count"], 3)]
        return {key: float(key == action) for key in shopping.ACTIONS}
    monkeypatch.setattr(shopping, "policy", policy)
    allocate, calls = population_profiles._budgeted_cart_units, []
    def budgeted(*args):
        calls.append(True)
        return allocate(*args)
    monkeypatch.setattr(population_profiles, "_budgeted_cart_units", budgeted)
    products = pd.DataFrame([dict(product_id=pid, category_id="fixture", unit_price_minor=price)
                             for pid, price in (("cheap", 500), ("target", 600))])
    result = EcommerceSimulator(config).generate(n_customers=8, months=6, seed=17, products=products,
                                               coupon_assignment={"mode": "none"})
    raw = {name: result.tables[name] for name in RAW_COLUMNS}
    validate_raw_dataset(raw, result.metadata)
    assert calls and len(raw["orders"]) > 0
    assert raw["orders"].checkout_source.eq("buy_now").all()
    assert raw["orders"].gross_minor.eq(600).all() and raw["orders"].paid_minor.eq(600).all()
    assert raw["order_items"].product_id.eq("target").all()
    assert not raw["events"].event_type.eq("remove_from_cart").any()
    cutoff = pd.Timestamp(result.metadata["observation_end_at"])
    added = raw["events"].loc[raw["events"].event_type.eq("cart")].groupby("customer_id").quantity.sum()
    assert all(cart_at(raw, uid, cutoff) == {"cheap": int(added[uid])} for uid in added.index)
