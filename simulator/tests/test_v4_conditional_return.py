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


def test_electronics_return_uses_exact_observed_gaussian_marginal():
    actions = ["view", "cart", "purchase"]
    active = [0, 1, 3, 4]
    marginal = np.array([[1., .2, .1, .3], [.2, 1.4, .1, -.2],
                         [.1, .1, .8, .15], [.3, -.2, .15, 1.2]])
    covariance = np.eye(5)
    covariance[np.ix_(active, active)] = marginal
    block = {"available_actions": actions, "mixture_weights": [.3, .7],
             "means": [[.1, -.2, 0., .5, 1.2], [1., .1, 0., .3, -.6]],
             "covariances": [covariance.tolist(), covariance.tolist()]}
    offsets = np.array([[.4, .2, 0., -.1]])

    def moments(value, source_offsets):
        recorded = []
        class Record:
            def choice(self, count, p):
                recorded.append(p.copy())
                return 0
            def normal(self, mean, scale):
                recorded.extend([mean, scale])
                return mean
        prepare_conditional_return(value, source_offsets, [0.], available_actions=actions)(0, Record())
        return recorded

    weights, mean, scale = moments(block, offsets)
    means = np.array(block["means"])
    delta = offsets[0, [0, 1, 3]] - means[:, [0, 1, 3]]
    inverse_delta = np.linalg.solve(marginal[:3, :3], delta.T).T
    expected_weights = np.array(block["mixture_weights"]) * np.exp(-.5 * (delta * inverse_delta).sum(axis=1))
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose(weights, expected_weights)
    assert mean == pytest.approx(means[0, 4] + marginal[3, :3] @ inverse_delta[0])
    assert scale == pytest.approx(np.sqrt(marginal[3, 3] - marginal[3, :3] @ np.linalg.solve(marginal[:3, :3], marginal[:3, 3])))

    block["means"][0][2], block["means"][1][2] = 1e8, -1e8
    offsets[0, 2] = np.nan
    for matrix in block["covariances"]:
        matrix[2][2] = -100  # The unobserved full covariance need not define the observed marginal.
        for index in active:
            matrix[2][index] = matrix[index][2] = 1e6
    changed = moments(block, offsets)
    np.testing.assert_array_equal(changed[0], weights)
    assert changed[1:] == [mean, scale]


def test_conditional_return_rejects_missing_or_conflicting_action_support():
    block = {"mixture_weights": [1.], "means": [np.zeros(5)], "covariances": [np.eye(5)]}
    with pytest.raises(ValueError, match="support differs"):
        prepare_conditional_return(block, [[0.] * 4], [0.], available_actions=["view", "cart", "purchase"])
    block["available_actions"] = ["view", "cart", "purchase"]
    with pytest.raises(ValueError, match="support differs"):
        prepare_conditional_return(block, [[0.] * 4], [0.], available_actions=["view", "cart", "remove", "purchase"])


def test_explicit_full_support_preserves_legacy_conditional_draw_stream():
    block = {"mixture_weights": [.25, .75], "means": [np.zeros(5), np.ones(5)],
             "covariances": [np.eye(5), np.eye(5)]}
    offsets = [[.2, -.1, .4, .1]]
    implicit = prepare_conditional_return(block, offsets, [0.])
    block["available_actions"] = ["view", "cart", "remove", "purchase"]
    explicit = prepare_conditional_return(block, offsets, [0.], available_actions=block["available_actions"])
    left, right = np.random.default_rng(92), np.random.default_rng(92)

    assert [implicit(0, left) for _ in range(20)] == [explicit(0, right) for _ in range(20)]
