import numpy as np
import pytest

from scripts.sweep_history_temperature import measures


def test_uniform_and_temperature_entropy():
    scores = np.tile(np.arange(27, dtype=float), (2, 1))
    actions = np.array([[0, 0, 0], [2, 2, 2]])
    cold = measures(scores, actions, 0.5)
    hot = measures(scores, actions, 2.0)
    assert hot['trajectory_entropy'] > cold['trajectory_entropy']
    assert all(np.isclose(sum(row), 1) for row in hot['predicted_action_shares_by_position'])
    assert measures(np.zeros((2, 27)), actions, 1)['trajectory_nll_per_action'] == pytest.approx(np.log(3))
    with pytest.raises(ValueError):
        measures(scores, actions, 0)
