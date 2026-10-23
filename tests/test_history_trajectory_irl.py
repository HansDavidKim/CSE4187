import numpy as np
import pandas as pd

from ecommerce_sim.history_trajectory_irl import QUADRATIC_NAMES, feature_counts, fit_maxent, evaluate
from scripts.run_history_trajectory_irl import windows_for_user


def test_history_is_prefix_only_and_remove_breaks_windows():
    frame = pd.DataFrame({'session': [1, 1, 1, 1, 1, 2, 2, 2],
                          'action': [0, 1, 3, 2, 0, 0, 1, 3]})
    h, p = windows_for_user(frame, 'cosmetics')
    assert h == [[0, 0, 0, 0, 0], [0, 0, 0, 1, 1]]
    assert p == [[0, 1, 2], [0, 1, 2]]


def test_candidate_features_follow_hypothetical_prefix():
    f = feature_counts(np.zeros((1, 5)), np.array([[0, 1, 2]]))[0, 0].reshape(2, 6)
    assert f[0, 0] == f[1, 0] == 1
    assert np.isclose(f[0, 1], np.log(2) / np.log(65))
    assert np.isclose(f[1, 2], np.log(2) / np.log(65))


def test_fit_and_probabilities():
    x = np.zeros((20, 5))
    y = np.tile([0, 1, 2], (20, 1))
    theta, _ = fit_maxent(x, y)
    metrics = evaluate(x, y, theta)
    assert np.isfinite(metrics['trajectory_nll_per_action'])
    assert np.isclose(sum(metrics['first_action_predicted_share']), 1)


def test_quadratic_includes_squares_and_pairs_but_no_cubic():
    assert len(QUADRATIC_NAMES) == 21
    assert len(set(QUADRATIC_NAMES)) == 21
    assert all(name.count('*') <= 1 for name in QUADRATIC_NAMES)
    initial = np.array([[3, 2, 0, 0, 0]])
    terms = feature_counts(initial, np.array([[1]]), quadratic=True)[0, 0].reshape(2, 21)[0]
    x, y = np.log1p(initial[0, :2]) / np.log(65)
    assert np.isclose(terms[QUADRATIC_NAMES.index('session_views^2')], x*x)
    assert np.isclose(terms[QUADRATIC_NAMES.index('session_views*session_carts')], x*y)
    assert np.allclose(feature_counts(initial, np.array([[1]]), quadratic=True)[0, 0].reshape(2, 21)[1], 0)


def test_l1_at_zero():
    x = np.zeros((10, 5))
    y = np.tile([0, 1, 2], (10, 1))
    theta, _ = fit_maxent(x, y, penalty=0, quadratic=True, l1=1000)
    assert np.allclose(theta, 0)
