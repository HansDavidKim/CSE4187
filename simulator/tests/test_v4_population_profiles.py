import json

import numpy as np

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
