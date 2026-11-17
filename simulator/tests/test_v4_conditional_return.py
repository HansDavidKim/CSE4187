import numpy as np
import pytest

from simulate.v4.irl.conditional_return import prepare_conditional_return


def test_gmm_return_draw_conditions_on_action_and_falls_back_outside_support():
    covariance = np.eye(5)
    covariance[4, 0] = covariance[0, 4] = 0.4
    block = {"mixture_weights": [0.25, 0.75], "means": [[0, 0, 0, 0, 2]] * 2,
             "covariances": [covariance, covariance]}
    draw = prepare_conditional_return(block, [[1, 0, 0, 0], [30, 0, 0, 0]], [0, -0.25])

    class RecordedDraw:
        def choice(self, count, p):
            assert count == 2 and p == pytest.approx([0.25, 0.75])
            return 0

        def normal(self, mean, scale):
            assert mean == pytest.approx(2.4)
            assert scale == pytest.approx(np.sqrt(1 - 0.4 ** 2))
            return mean

    assert draw(0, RecordedDraw()) == pytest.approx(2.4)
    assert draw(1, None) == -0.25
