import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl.fit_checkout_spending import (
    TYPE, fit_cart_fulfillment, fit_spending_model, spending_pool_weights, validate_spending_model,
)
from simulate.v4.simulator.population_profiles import make_catalog, prepare_population


def source_episodes():
    rng = np.random.default_rng(7)
    rows = []
    for user in range(80):
        preference = rng.normal([1.5, 6.], [.8, .7])
        for number in range(4):
            values = preference + rng.normal(0, [.3, .25])
            rows.append(dict(episode_id=len(rows), user_id=str(user),
                at=pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=14 * number),
                unit_count=max(1, int(np.exp(values[0]))), unit_mean_price_minor=np.exp(values[1]),
                catalog_supported=True))
    return pd.DataFrame(rows)


def purchase_rows(episodes):
    return pd.DataFrame({"user_id": episodes.user_id, "at": episodes["at"],
                         "action": "purchase", "price_minor": episodes.unit_mean_price_minor,
                         "product_id": "sku", "session": "fixture"})


def test_fit_separates_persistent_variation_and_excludes_heldout_targets():
    episodes = source_episodes()
    purchases = purchase_rows(episodes)
    fitted = fit_spending_model(episodes, "2020-02-01T00:00:00Z", purchase_rows=purchases)
    assert fitted["fit"]["repeat_training_users"] > 30
    for split in fitted["validation"].values():
        assert split["records"] and split["persistent_nll"] < split["independent_nll"]
    heldout = episodes["at"].ge("2020-02-01T00:00:00Z") | episodes.user_id.map(
        lambda uid: hashlib.sha256(f"42:{uid}".encode()).digest()[0] >= 205)
    episodes.loc[heldout, "unit_mean_price_minor"] *= 100
    purchases.loc[heldout, "price_minor"] *= 100
    changed = fit_spending_model(episodes, "2020-02-01T00:00:00Z", purchase_rows=purchases)
    assert fitted["assortment_price_copula"]["rho"] > .5
    for name in ("mean", "within_covariance", "between_covariance", "fit", "assortment_price_copula"):
        assert changed[name] == fitted[name]


def test_density_ratio_preserves_marginal_and_does_not_invent_support():
    model = {"schema_version": 1, "type": TYPE, "mean": [0., 0.],
             "within_covariance": [[1., 0.], [0., 1.]], "between_covariance": [[0., 0.], [0., 0.]]}
    weights = np.array([1., 2., 0.])
    np.testing.assert_allclose(spending_pool_weights(model, weights, [1, 2, 3], [1, 3, 100], [0, 0]), [1/3, 2/3, 0])
    model["between_covariance"] = [[1., 0.], [0., 1.]]
    low = spending_pool_weights(model, weights, [1, 2, 3], [1, 3, 100], [-1, -1])
    high = spending_pool_weights(model, weights, [1, 2, 3], [1, 3, 100], [2, 2])
    assert high[1] > low[1] and high[2] == low[2] == 0
    with pytest.raises(ValueError):
        spending_pool_weights(model, weights, [0, 2, 3], [1, 3, 100], [0, 0])
    for invalid in ([[1, 2], [2, 1]], [[1, 0], [0, 0]], [[1, 0], [0, float("nan")]]):
        changed = copy.deepcopy(model)
        changed["within_covariance"] = invalid
        with pytest.raises(ValueError):
            validate_spending_model(changed)


def test_bundled_population_has_private_reproducible_persistent_spending_draw():
    config = Path(__file__).parents[1] / "src/simulate/v4/config"
    shopping = json.loads((config / "population_shopping.json").read_bytes())
    population = shopping["population"]
    def draw():
        rng = np.random.default_rng(42)
        products = make_catalog(population, rng)
        _, profile = prepare_population(population, products, rng)
        return [profile(i) for i in range(3)]
    profiles = draw()
    assert profiles == draw()
    assert len({tuple(profile["checkout_spending_preference"]) for profile in profiles}) == 3
    gmm = json.loads((config / "population_gmm.json").read_bytes())
    assert gmm["base_model_sha256"] == hashlib.sha256((config / "population_shopping.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("binding", ["contract", "source", "validation"])
@pytest.mark.filterwarnings("ignore:.*found in sys.modules.*:RuntimeWarning")
def test_cli_uses_exporter_training_cutoff_precedence(tmp_path, monkeypatch, binding):
    from simulate.v4.irl import fit_checkout_transition, fit_donor_prior
    source = {"catalog": [], "seed": 42, "validation": {"cutoff_at_utc": "2020-02-01T00:00:00Z"},
              "source": {"training_cutoff_at": "2020-01-29T00:00:00Z"}}
    if binding == "contract":
        source["source_contract"] = {"training_cutoff_at": "2020-01-28T00:00:00Z"}
    elif binding == "validation":
        source["source"] = {}
    model, output = tmp_path / "source.json", tmp_path / "spending.json"
    model.write_text(json.dumps(source))
    monkeypatch.setattr(fit_donor_prior, "load_prior_source", lambda *_: (purchase_rows(source_episodes()), {}))
    monkeypatch.setattr(fit_checkout_transition, "build_checkout_episodes", lambda *_: source_episodes())
    monkeypatch.setattr(sys, "argv", ["fit_checkout_spending", "--source-model", str(model),
        "--source-cache", str(tmp_path / "unused.parquet"), "--output", str(output)])
    runpy.run_module("simulate.v4.irl.fit_checkout_spending", run_name="__main__")
    expected = {"contract": "2020-01-28", "source": "2020-01-29", "validation": "2020-02-01"}[binding]
    assert json.loads(output.read_text())["fit"]["cutoff_at"] == f"{expected}T00:00:00+00:00"


def test_assortment_copula_preserves_group_price_and_joint_spending_marginals():
    spending = {"schema_version": 1, "type": TYPE, "mean": [1., 6.],
        "between_covariance": [[.3, -.1], [-.1, .4]], "within_covariance": [[.5, 0.], [0., .5]],
        "assortment_price_copula": {"type": "source_gaussian_rank_preferred_checkout_price", "rho": .8}}
    population = {"schema_version": 1, "catalog_bands": [[[100, 1, 1], [200, 1, 1], [500, 1, 1], [1000, 1, 1], [2000, 1, 1]]],
        "checkout_counts": [[1, 1, 500, 1]], "checkout_spending": spending,
        "home_groups": {"buyer": {"customers": 1, "component_probabilities": [1.],
            "visit_probability_30d": .5, "leading_preference_share": 1., "checkout_preference_share": 0.,
            "choice_gap_counts": [[0, 1]], "assortment_size_counts": [[1, 1]],
            "preferred_price_counts": [[500, 4], [100, 1], [2000, 1], [200, 2], [1000, 2]]}}}
    rng = np.random.default_rng(81)
    _, draw = prepare_population(population, make_catalog(population, rng), rng)
    profiles = [draw(n) for n in range(12000)]
    prices = pd.Series([p["preferred_price_minor"] for p in profiles])
    np.testing.assert_allclose(prices.value_counts(normalize=True).sort_index(), [.1, .2, .4, .2, .1], atol=.015)
    preferences = np.array([p["checkout_spending_preference"] for p in profiles])
    np.testing.assert_allclose(preferences.mean(axis=0), spending["mean"], atol=.02)
    np.testing.assert_allclose(np.cov(preferences.T), spending["between_covariance"], atol=.02)
    assert .7 < prices.corr(pd.Series(preferences[:, 1]), method="spearman") < .85
    invalid = copy.deepcopy(spending)
    invalid["assortment_price_copula"]["rho"] = 1.1
    with pytest.raises(ValueError, match="copula"):
        validate_spending_model(invalid)


def test_cart_fulfillment_replays_exact_known_inventory_and_excludes_heldouts():
    uid = next(str(n) for n in range(100) if hashlib.sha256(f"42:{n}".encode()).digest()[0] < 205)
    heldout_uid = next(str(n) for n in range(100) if hashlib.sha256(f"42:{n}".encode()).digest()[0] >= 205)
    actions = [(0, 'cart', 'A'), (1, 'cart', 'A'), (2, 'cart', 'B'),
               (3, 'purchase', 'A'), (3, 'purchase', 'B'), (4, 'cart', 'B'),
               (5, 'purchase', 'A'), (5, 'purchase', 'Y'), (5, 'purchase', 'Y'),
               (6, 'purchase', 'Y'), (7, 'remove', 'B'), (8, 'remove', 'B'), (9, 'purchase', 'Y')]
    rows = pd.DataFrame([{'user_id': uid, 'at': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(seconds=at),
        'action': action, 'product_id': product, 'session': 'fixture', 'source_event_id': f'{n:04}'}
        for n, (at, action, product) in enumerate(actions)])
    fitted = fit_cart_fulfillment(rows, '2020-02-01T00:00:00Z')
    assert fitted['fraction_counts'] == [[0, 1, 1], [1, 2, 1], [1, 1, 1]]
    assert fitted['fit'] == {'training_checkouts': 4, 'purchase_unit_proxies': 7, 'matched_cart_unit_proxies': 3,
        'checkouts_with_known_cart_capacity': 3, 'known_cart_capacity_unit_proxies': 5,
        'remove_rows_without_known_unit': 1}
    future = rows.copy(); future['at'] += pd.Timedelta(days=90)
    heldout = rows.copy(); heldout['user_id'] = heldout_uid
    assert fit_cart_fulfillment(pd.concat([rows, future, heldout]), '2020-02-01T00:00:00Z') == fitted
    assert fit_cart_fulfillment(rows.loc[rows.action.eq('purchase')], '2020-02-01T00:00:00Z') is None


def test_cart_fulfillment_keeps_same_timestamp_nonpurchase_separator():
    uid = next(str(n) for n in range(100) if hashlib.sha256(f"42:{n}".encode()).digest()[0] < 205)
    rows = pd.DataFrame([{'user_id': uid, 'at': pd.Timestamp('2020-01-01', tz='UTC') + pd.Timedelta(seconds=at),
        'action': action, 'product_id': product, 'session': 'fixture', 'source_event_id': f'{n:04}'}
        for n, (at, action, product) in enumerate([(0, 'cart', 'A'), (0, 'cart', 'B'),
            (1, 'purchase', 'A'), (1, 'view', 'A'), (1, 'purchase', 'B')])])
    fitted = fit_cart_fulfillment(rows, '2020-02-01T00:00:00Z')
    assert fitted['fraction_counts'] == [[1, 1, 2]] and fitted['fit']['training_checkouts'] == 2


@pytest.mark.parametrize('fractions', [[[True, 1, 1]], [[1, 0, 1]], [[2, 1, 1]], [[0, 1, 0]],
    [[1, 2, 1], [2, 4, 1]], []])
def test_cart_fulfillment_rejects_invalid_fraction_counts(fractions):
    model = {'schema_version': 1, 'type': TYPE, 'mean': [0., 0.],
        'within_covariance': [[1., 0.], [0., 1.]], 'between_covariance': [[0., 0.], [0., 0.]],
        'cart_fulfillment': {'type': 'source_observed_checkout_cart_fraction', 'fraction_counts': fractions}}
    with pytest.raises(ValueError, match='cart fulfillment'):
        validate_spending_model(model)
