import numpy as np
import pytest

from models.conditional_trajectory_irl import fit_reward, offer_transition, probabilities, sample_coefficients


def test_offer_changes_purchase_probability_not_the_action_space():
    p = probabilities(np.array([[0., 0., 2., 0.]]), np.eye(2), 1)
    assert p.shape == (1, 1, 2, 2, 2)
    assert np.isclose(p[0, 0, 0, 0, 1], .5)
    assert p[0, 0, 0, 1, 1] > .85
    assert np.allclose(p.sum(axis=-1), 1)


def test_treatment_fit_recovers_direction_and_checks_trajectory():
    rng = np.random.default_rng(9)
    offer = rng.integers(0, 2, (1000, 1))
    action = (rng.random((1000, 1)) < np.where(offer, .8, .2)).astype(int)
    states = np.zeros_like(offer)
    weights, _ = fit_reward(states, offer, action, np.full((2, 2), .5), row_group=np.zeros(1000, dtype=int), penalty=1)
    assert weights[0, 2] > 1
    with pytest.raises(ValueError):
        fit_reward(np.array([[0, 1]]), np.array([[0, 0]]), np.array([[0, 1]]), np.eye(2))


def test_offer_transition_normalized():
    transition = offer_transition(np.array([[0, 0, 1, 1], [1, 0, 0, 1]]))
    assert np.allclose(transition.sum(axis=1), 1)
    assert (transition > 0).all()


def test_coefficient_mixture_sampling_is_seeded():
    weights, group = sample_coefficients(np.array([1.]), np.zeros((1, 4)), np.eye(4)[None], 20, 4)
    again, group_again = sample_coefficients(np.array([1.]), np.zeros((1, 4)), np.eye(4)[None], 20, 4)
    assert weights.shape == (20, 4)
    assert np.array_equal(weights, again)
    assert np.array_equal(group, group_again)
