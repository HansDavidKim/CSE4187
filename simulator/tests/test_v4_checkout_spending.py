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
    TYPE, fit_spending_model, spending_pool_weights, validate_spending_model,
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


def test_fit_separates_persistent_variation_and_excludes_heldout_targets():
    episodes = source_episodes()
    fitted = fit_spending_model(episodes, "2020-02-01T00:00:00Z")
    assert fitted["fit"]["repeat_training_users"] > 30
    for split in fitted["validation"].values():
        assert split["records"] and split["persistent_nll"] < split["independent_nll"]
    heldout = episodes["at"].ge("2020-02-01T00:00:00Z") | episodes.user_id.map(
        lambda uid: hashlib.sha256(f"42:{uid}".encode()).digest()[0] >= 205)
    episodes.loc[heldout, "unit_mean_price_minor"] *= 100
    changed = fit_spending_model(episodes, "2020-02-01T00:00:00Z")
    for name in ("mean", "within_covariance", "between_covariance", "fit"):
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
    monkeypatch.setattr(fit_donor_prior, "load_prior_source", lambda *_: (pd.DataFrame(), {}))
    monkeypatch.setattr(fit_checkout_transition, "build_checkout_episodes", lambda *_: source_episodes())
    monkeypatch.setattr(sys, "argv", ["fit_checkout_spending", "--source-model", str(model),
        "--source-cache", str(tmp_path / "unused.parquet"), "--output", str(output)])
    runpy.run_module("simulate.v4.irl.fit_checkout_spending", run_name="__main__")
    expected = {"contract": "2020-01-28", "source": "2020-01-29", "validation": "2020-02-01"}[binding]
    assert json.loads(output.read_text())["fit"]["cutoff_at"] == f"{expected}T00:00:00+00:00"
