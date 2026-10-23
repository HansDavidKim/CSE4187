import numpy as np
import pandas as pd

from ecommerce_sim.uplift import FEATURES, StarbucksUpliftSimulator


def sample(rows=1200, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, len(FEATURES)))
    action = rng.binomial(1, 0.5, rows)
    probability = 1 / (1 + np.exp(-(-3.5 + 0.4 * x[:, 0] + action * (0.8 + 0.5 * x[:, 1]))))
    frame = pd.DataFrame(x, columns=FEATURES)
    frame.insert(0, "purchase", rng.binomial(1, probability))
    frame.insert(0, "Promotion", np.where(action, "Yes", "No"))
    frame.insert(0, "ID", np.arange(rows))
    return frame


def test_uplift_simulator(tmp_path):
    train, test = sample(2000), sample(800, seed=8)
    simulator = StarbucksUpliftSimulator(segments=3).fit(train)
    p0, p1 = simulator.potential_probabilities(test)
    assert np.mean(p1 - p0) > 0
    metrics = simulator.evaluate(test)
    assert metrics["target_fraction"] == 0.30
    generated = simulator.generate(1000, policy="targeted")
    assert len(generated) == 1000
    assert set(generated.Promotion) <= {"Yes", "No"}
    assert generated.purchase.isin([0, 1]).all()
    for feature in FEATURES:
        assert generated[feature].between(train[feature].min(), train[feature].max()).all()
    simulator.save(tmp_path, test, synthetic_rows=100)
    assert (tmp_path / "uplift_metrics.json").exists()

