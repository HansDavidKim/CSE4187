import numpy as np
import pytest

from ecommerce_sim.common_behavior_irl import COEFFICIENT_NAMES, trajectory_windows


def test_windows_respect_boundaries_and_action_mapping():
    states, actions = trajectory_windows([0, 1, 3, 0, 2, 0, 1, 3, 0, 4],
                                         ['view', 'cart', 'remove', 'purchase'], 3)
    assert states.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert actions.tolist() == [[1, 2, 0], [1, 2, 0]]
    assert len(COEFFICIENT_NAMES) == 8
    assert np.array_equal(states[:, 1:], actions[:, :-1])


def test_no_fabricated_transition_across_unsupported_event():
    states, actions = trajectory_windows([0, 1, 2, 0, 1, 3, 4],
                                         ['view', 'cart', 'remove', 'purchase'], 3)
    assert states.shape == actions.shape == (0, 3)
    with pytest.raises(ValueError):
        trajectory_windows([], ['view'], 0)
