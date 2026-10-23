import numpy as np
import pytest

from ecommerce_sim.user_choice import UserMaxEnt, make_states, state_spec, weekly_choices


def test_weekly_choice_labels_and_impossible_discount():
    data = np.array([[0, 0], [2, 0], [1, 3]])
    assert weekly_choices(data, ['transactions', 'coupon_discount']).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        weekly_choices(np.array([[0, 3]]), ['transactions', 'coupon_discount'])


def test_state_uses_lagged_parents_and_never_current_outcome():
    graph = {'converged': True, 'instantaneous_is_dag': True, 'edges': [
        {'source': 'quantity', 'target': 'transactions', 'lag': 1},
        {'source': 'quantity', 'target': 'transactions', 'lag': 0}]}
    spec = state_spec(graph)
    assert ('quantity', 1) in spec and ('quantity', 0) not in spec
    x = np.arange(4 * 3).reshape(1, 4, 3)
    states = make_states(x, ['transactions', 'active_offers', 'quantity'], spec, np.array([2]))
    assert states[0, 0, spec.index(('quantity', 1))] == x[0, 1, 2]


def test_user_reward_is_per_user_and_utilities_are_finite():
    rng = np.random.default_rng(7)
    states = rng.normal(size=(4, 40, 2))
    actions = rng.integers(0, 3, size=(4, 40))
    actions[0, :20] = 0
    actions[1, :20] = 2
    model = UserMaxEnt(2).fit(states, actions)
    assert model.weights_.shape == (4, 2, 3)
    assert not np.allclose(model.weights_[0], model.weights_[1])
    assert np.isfinite(model.utilities(states)).all()
    assert (model.utilities(states)[..., 0] == 0).all()
