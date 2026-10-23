import numpy as np
from ecommerce_sim.latent_session_model import (LatentSessionUser, candidate_paths, coupon_outcomes, fit_reward,
    path_features, path_probabilities, sequential_choice)


def test_path_features_and_soft_conditionals():
    paths = candidate_paths(3)
    assert len(paths) == 39
    assert path_features((0, 1), [.2, .8], 3).shape == (15,)
    w = np.zeros(15)
    _, p = path_probabilities([[.2, .8]], w, 3)
    assert np.isclose(p.sum(), 1)
    first = sequential_choice(paths, p[0], ())
    assert set(first) == {0, 1, 2}
    later = sequential_choice(paths, p[0], (0,))
    assert 'END' in later and np.isclose(sum(later.values()), 1)
    assert sequential_choice(paths, p[0], (0, 1, 2)) == {'END': 1.0}


def test_fit_and_scenario():
    observed = [(0,), (0, 1), (1,), (0, 0)] * 5
    z = np.tile([.4, .6], (len(observed), 1))
    w, diag = fit_reward(z, observed, 2, l1=.1)
    assert np.isfinite(w).all() and diag['iterations'] > 0
    assert coupon_outcomes(.2, .6, .4) == (False, True)


def test_session_user_preserves_unobserved_tail():
    class Scaler:
        def transform(self, frame):
            return frame.to_numpy()
    class GMM:
        def predict_proba(self, frame):
            return np.array([[.4, .3, .3]])
    class NeverReturn:
        def predict_proba(self, frame):
            return np.array([[1., 0.]])
    model = {'actions': ['view', 'cart', 'purchase'], 'prefix_cols': ['previous_gap_hours'],
             'scaler': Scaler(), 'gmm': GMM(), 'reward': np.zeros(15), 'hazard': NeverReturn(),
             'hazard_edges_hours': np.array([0., 1., 720.])}
    user = LatentSessionUser(model, {'previous_gap_hours': .1}, np.random.default_rng(0))
    events = user.sample_session()
    assert 1 <= len(events) <= 3
    assert user.finish_session(events) is None
    assert user.completed_sessions == 3
