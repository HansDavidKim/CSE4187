import numpy as np
from analysis.persona.behavior import action_bins, delay_bins, qvalues


def test_action_bin_boundaries():
    assert action_bins([0, 1, 2, 3, 4], [4]*5) == [1, 1, 1, 2]


def test_delay_bins_include_censored_and_outside_horizon():
    assert delay_bins([0, .5, 24, 168, 720, 721, np.nan]) == [2, 1, 1, 1, 2]
    assert qvalues([]) is None
