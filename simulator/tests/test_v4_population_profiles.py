import json

import numpy as np
import pytest

from simulate.v4.simulator.population_profiles import (
    aggregate_population, align_checkout_basket, make_catalog, prepare_population,
)


def test_population_generates_reproducible_profiles_and_baskets_without_source_ids():
    products = [{"product_id": "original-sku", "category_id": "original-category", "unit_price_minor": 1000}]
    profiles = [{"profile_id": "original-customer", "home_group": "buyer", "preferred_price_minor": 1000,
                 "product_weights": {"original-sku": 3}, "visit_probability_30d": 0.4, "choice_gap_seconds": [30]}]
    basket = {"items": [{"product_id": "original-sku", "quantity": 3}], "observed_count": 8}
    profiles[0]["checkout_templates"] = [basket]
    latent = {"mixture_weights": [0.5, 0.5], "profile_posteriors": {"original-customer": [0.2, 0.8]}}
    calibration = aggregate_population({"catalog": products, "profiles": profiles, "latent_shopping": latent,
                                        "distributions": {"checkout_templates": [basket]}})
    assert "original-" not in json.dumps(calibration)

    def generated():
        rng = np.random.default_rng(42)
        catalog = make_catalog(calibration, rng)
        groups, sample = prepare_population(calibration, catalog, rng)
        return catalog, groups, [sample(i) for i in range(2)]

    catalog, groups, profiles = generated()
    assert (catalog, groups, profiles) == generated()
    sku = catalog[0]["product_id"]
    assert {profile["profile_id"] for profile in profiles} == {"SYN000000000", "SYN000000001"}
    assert all(profile["component_probabilities"] == [0.2, 0.8] and profile["product_weights"] == {sku: 1}
               and profile["checkout_preference_share"] == 1.0 for profile in profiles)
    templates = groups["buyer"]["checkout_templates"]
    assert all(template["items"] == [{"product_id": sku, "quantity": 3}] for template in templates)
    assert sum(template["observed_count"] for template in templates) == 8


def test_population_checkout_reuses_favorites_at_the_selected_price():
    products = {sku: {"unit_price_minor": price}
                for sku, price in {"A": 100, "B": 100, "C": 100, "D": 500, "E": 505}.items()}
    profile = {"product_weights": {"B": 3, "C": 2, "E": 1}, "checkout_preference_share": 1.0}
    basket = {"A": 2, "B": 4, "D": 1}
    aligned = align_checkout_basket(basket, profile, products, np.random.default_rng(42))
    assert aligned == {"C": 2, "B": 4, "D": 1}
    assert basket == {"A": 2, "B": 4, "D": 1}


def test_checkout_prefers_held_units_and_merges_shared_replacements():
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"template-a": 100, "template-b": 100, "held-a": 102, "held-b": 105, "other": 500}.items()}
    basket, cart = {"template-a": 2, "template-b": 2}, {"held-a": 3, "held-b": 1, "other": 5}
    result = align_checkout_basket(basket, {"checkout_preference_share": 0}, products,
                                    np.random.default_rng(42), cart=cart)
    assert result == {"held-a": 3, "held-b": 1}
    assert sum(result.values()) == sum(basket.values())
    assert cart == {"held-a": 3, "held-b": 1, "other": 5}
    assert basket == {"template-a": 2, "template-b": 2}


def test_checkout_keeps_unmatched_targets_and_uses_only_available_cart_units():
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"target": 100, "same-price": 100, "other": 500}.items()}
    result = align_checkout_basket({"target": 3}, {}, products, np.random.default_rng(42),
                                    cart={"same-price": 1, "other": 5})
    assert result == {"same-price": 1, "target": 2}
    assert sum(products[sku]["unit_price_minor"] * qty for sku, qty in result.items()) == 300
    assert align_checkout_basket({"target": 1}, {}, products, np.random.default_rng(42),
                                  cart={"target": 1, "same-price": 2}) == {"target": 1}


def test_source_cart_quota_uses_held_goods_before_direct_template_units():
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"low-target": 300, "high-target": 900, "held": 600, "unbought": 2000}.items()}
    basket, cart = {"low-target": 1, "high-target": 1}, {"held": 3, "unbought": 4}
    policy = {"fraction_counts": [[1, 1, 1]]}
    actual = align_checkout_basket(basket, {}, products, np.random.default_rng(42),
                                  cart=cart, cart_fulfillment=policy)
    assert actual == {"held": 2}
    assert sum(actual.values()) == 2
    assert sum(products[p]["unit_price_minor"] * n for p, n in actual.items()) == 1200
    assert cart == {"held": 3, "unbought": 4} and basket == {"low-target": 1, "high-target": 1}


def test_cart_quota_preserves_existing_matches_and_bounded_inventory():
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"target": 100, "held": 105, "existing": 500}.items()}
    for fraction, expected in (([0, 1, 1], {"target": 3, "existing": 1}),
                               ([1, 1, 1], {"target": 1, "held": 2, "existing": 1})):
        actual = align_checkout_basket({"target": 3, "existing": 1}, {}, products, np.random.default_rng(42),
            cart={"held": 2, "existing": 1}, cart_fulfillment={"fraction_counts": [fraction]})
        assert actual == expected and sum(actual.values()) == 4
    rng = np.random.default_rng(42)
    draws = [align_checkout_basket({"target": 1}, {}, products, rng,
        cart={"held": 1}, cart_fulfillment={"fraction_counts": [[1, 2, 1]]}) for _ in range(2000)]
    assert abs(sum(row.get("held", 0) for row in draws) / len(draws) - .5) < .04


def test_checkout_balances_held_prices_inside_whole_order_budget():
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"target": 600, "cheap": 500, "dear": 700, "unaffordable": 5000}.items()}
    policy = {"fraction_counts": [[1, 1, 1]]}
    actual = align_checkout_basket({"target": 4}, {}, products, np.random.default_rng(42),
        cart={"cheap": 4, "dear": 2}, cart_fulfillment=policy)
    assert actual == {"cheap": 2, "dear": 2}
    assert align_checkout_basket({"target": 1}, {}, products, np.random.default_rng(42),
        cart={"unaffordable": 3}, cart_fulfillment=policy) == {"target": 1}


def test_cart_budget_rejects_unverified_fractional_solver_allocation(monkeypatch):
    from types import SimpleNamespace
    from scipy import optimize
    products = {sku: {"unit_price_minor": price} for sku, price in
                {"target": 600, "cheap": 500, "dear": 700}.items()}
    monkeypatch.setattr(optimize, "milp", lambda *a, **kw: SimpleNamespace(success=True, x=np.array([.5, 1.5, 2.])))
    with pytest.raises(ValueError, match="violates checkout quantities or budget"):
        align_checkout_basket({"target": 4}, {}, products, np.random.default_rng(42),
            cart={"cheap": 4, "dear": 2}, cart_fulfillment={"fraction_counts": [[1, 1, 1]]})


def test_direct_favorites_cannot_bypass_observed_cart_quota():
    products = {pid: {"unit_price_minor": 100} for pid in ("target", "held")}
    assert align_checkout_basket({"target": 1},
        {"product_weights": {"held": 1}, "checkout_preference_share": 1}, products,
        np.random.default_rng(42), cart={"held": 1},
        cart_fulfillment={"fraction_counts": [[0, 1, 1]]}) == {"target": 1}
