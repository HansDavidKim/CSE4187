import numpy as np

from ecommerce_sim.choice import CalibratedMultinomialLogit


def test_calibrated_multinomial_logit_is_stochastic_and_normalized():
    utilities = np.array([
        [2.0, 0.0, -1.0],
        [0.0, 2.0, -1.0],
        [1.0, 0.5, 0.0],
        [0.2, 0.5, 1.0],
    ])
    choices = np.array([0, 1, 0, 2])
    model = CalibratedMultinomialLogit(min_temperature=0.2, max_temperature=3.0).fit(
        utilities, choices
    )
    probability = model.probabilities(utilities)
    diagnostics = model.diagnostics(utilities)
    assert np.allclose(probability.sum(axis=1), 1)
    assert np.all((probability > 0) & (probability < 1))
    assert 0.2 <= diagnostics["temperature"] <= 3.0
    assert 0 < diagnostics["mean_normalized_entropy"] < 1
    assert diagnostics["mean_top1_probability"] > 1 / 3
