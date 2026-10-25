import numpy as np
from simulate.latent_session_model import (LatentSessionUser, candidate_paths, coupon_outcomes, fit_reward,
    path_features, path_probabilities, persona_gap_features, sample_persona_gap, sequential_choice)
from simulate.latent_session_model import (
    canonical_type_probabilities, churn_sequence_features, coupon_action_probabilities, coupon_offsets,
    lifecycle_action_probabilities, sample_internal_gap, sample_home_persona,
    sample_lifecycle_step, sample_new_user, sample_reward_coefficients,
    simulate_no_uplift_users,
    update_gap_internal_state,
)


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


def test_persona_gap_sampler_uses_past_session_state():
    state = {'view': 2, 'cart': 1, 'remove': 0, 'purchase': 0,
             'duration_hours': .5, 'events': 3, 'previous_gap_hours': 24}
    assert persona_gap_features(state).shape == (9,)
    model = {'edges_hours': [0, 1, 6], 'personas': {'0': {
        'mean': [0] * 9, 'scale': [1] * 9,
        'coefficient': [30, -30] + [0] * 9,
    }}}
    gap = sample_persona_gap(model, 0, state, np.random.default_rng(3))
    assert 0 <= gap <= 1
    missing = dict(state, previous_gap_hours=None, initial_gap_hours=48)
    assert np.isclose(persona_gap_features(missing)[-2], np.log1p(48))
    assert persona_gap_features(missing)[-1] == 0


def test_internal_gap_state_updates_and_samples_constrained_token():
    summary = {'view': 4, 'cart': 2, 'remove': 1, 'purchase': 1,
               'events': 8, 'previous_gap_hours': 12}
    signal, state = update_gap_internal_state(np.zeros(6), summary, .5, 48)
    assert np.all((0 <= signal) & (signal <= 1))
    assert np.allclose(state, signal / 2)
    model = {
        'initial_gap_hours': 48,
        'edges_hours': [0, 1, 6, 24, 72, 168, 336, 720],
        'gap_tokens': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'none'],
        'personas': {'0': {
            'memory': .5, 'features': 'internal_state',
            'mean': [0] * 6, 'scale': [1] * 6,
            'coefficient': [[0] * 6 for _ in range(7)],
            'intercept': [-30, 30, -30, -30, -30, -30, -30],
            'churn_features': 'internal_state',
            'churn_mean': [0] * 6, 'churn_scale': [1] * 6,
            'churn_coefficient': [0] * 6, 'churn_intercept': -30,
        }},
    }
    gap, updated, token = sample_internal_gap(
        model, 0, np.zeros(6), summary, np.random.default_rng(4)
    )
    assert token == 'b' and 1 <= gap <= 6
    assert np.allclose(updated, state)


def test_churn_sequence_features_keep_two_most_recent_core_signals():
    current = np.arange(6)
    history = [np.arange(4) + 10, np.arange(4) + 20, np.arange(4) + 30]
    features = churn_sequence_features(current, history)
    assert features.shape == (14,)
    assert np.array_equal(features[:4], current[:4])
    assert np.array_equal(features[4:8], history[-1])
    assert np.array_equal(features[8:12], history[-2])
    assert np.array_equal(features[-2:], [1, 1])


def test_new_user_keeps_home_persona_across_churn_and_recovery():
    specification = {
        'memory': 0., 'features': 'internal_state',
        'mean': [0] * 6, 'scale': [1] * 6,
        'coefficient': [[0] * 6 for _ in range(7)], 'intercept': [30] + [-30] * 6,
        'churn_features': 'internal_state',
        'churn_mean': [0] * 6, 'churn_scale': [1] * 6,
        'churn_coefficient': [0] * 6, 'churn_intercept': -30,
    }
    model = {
        'initial_gap_hours': 0,
        'edges_hours': [0, 1, 6, 24, 72, 168, 336, 720],
        'gap_tokens': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'NO_RETURN_30D'],
        'personas': {'0': specification, '2': specification, '3': specification},
        'lifecycle': {
            'mature_personas': [0, 2, 3],
            'home_persona_prior': {'0': 0., '2': 1., '3': 0.},
            'onboarding_sessions': 2,
        },
    }
    rng = np.random.default_rng(5)
    user = sample_new_user(model, rng)
    assert user == {'home_persona': 2, 'lifecycle_state': 'NEW'}
    assert sample_home_persona(model, rng) == 2
    result = sample_lifecycle_step(
        model, 2, 'NEW', np.zeros(6), {'events': 1}, rng, completed_sessions=2
    )
    assert result[3] == 'P2'
    result = sample_lifecycle_step(
        model, 2, 'CHURNING', np.zeros(6), {'events': 1}, rng
    )
    assert result[3] == 'P2'


def test_purchase_then_fitted_nonreturn_creates_mission_exit():
    specification = {
        'memory': 0., 'features': 'internal_state',
        'mean': [0] * 6, 'scale': [1] * 6,
        'coefficient': [[0] * 6 for _ in range(7)], 'intercept': [0] * 7,
        'churn_features': 'internal_state',
        'churn_mean': [0] * 6, 'churn_scale': [1] * 6,
        'churn_coefficient': [0] * 6, 'churn_intercept': 30,
    }
    model = {
        'initial_gap_hours': 0, 'edges_hours': [0, 1, 6, 24, 72, 168, 336, 720],
        'gap_tokens': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'NO_RETURN_30D'],
        'personas': {'0': specification},
        'lifecycle': {'mature_personas': [0], 'home_persona_prior': {'0': 1.}},
    }
    result = sample_lifecycle_step(
        model, 0, 'P0', np.zeros(6), {'events': 1, 'purchase': 1},
        np.random.default_rng(2),
    )
    assert result[0] is None and result[2:] == ('NO_RETURN_30D', 'MISSION_EXIT')


def test_reward_coefficient_sampler_reconstructs_identification_constraint():
    model = {
        'weights': [1.], 'standardized_means': [[1., 2.]],
        'standardized_variances': [[0., 0.]], 'scaler_mean': [0., 0.],
        'scaler_scale': [1., 1.], 'terms': ['cart:x', 'view:x'],
        'dependent_action': 'end',
        'component_persona_composition': [{'gmm_component': 0, 'persona': 2, 'size': 10}],
    }
    component, coefficients = sample_reward_coefficients(model, np.random.default_rng(2))
    assert component == 0
    assert coefficients == {'cart': {'x': 1.}, 'view': {'x': 2.}, 'end': {'x': -3.}}
    assert sample_reward_coefficients(model, np.random.default_rng(2), persona=2)[0] == 0


def test_lifecycle_state_changes_immediate_action_probability():
    theta = np.zeros((5, 19))
    theta[3, 14] = 2.0  # purchase reward responds to prior satiation
    model = {
        'mean': [0.] * 18, 'scale': [1.] * 18,
        'actions': ['view', 'cart', 'remove', 'purchase', 'end'],
        'features': [f'f{i}' for i in range(19)],
        'personas': {'0': {'coefficient': theta.tolist()}},
    }
    action = np.r_[1., np.zeros(11)]
    low = lifecycle_action_probabilities(model, 0, action, np.zeros(6), False)
    state = np.zeros(6); state[2] = 1
    high = lifecycle_action_probabilities(model, 0, action, state, False)
    assert np.isclose(low.sum(), 1) and np.isclose(high.sum(), 1)
    assert high[3] > low[3]


def test_coupon_offsets_decay_stack_and_preserve_zero_uplift_policy():
    model = {
        'type_multipliers': {
            'sure_thing': 0., 'persuadable': 1.,
            'sleeping_dog': -1., 'lost_cause': 0.,
        },
        'decay': {'rho': 2., 'horizon_days': 42},
        'default_strength': 1., 'strength_by_persona': {'0': 2.},
        'personas': {'0': {'purchase_logit_shift': 1., 'return_logit_shift': .5}},
    }
    base = np.array([.4, .2, .1, .1, .2])
    assert coupon_offsets(model, 0, 'persuadable', []) == {'purchase': 0., 'return': 0.}
    assert np.allclose(
        coupon_action_probabilities(base, model, 0, 'persuadable', []), base
    )
    one = coupon_offsets(model, 0, 'persuadable', [0])['purchase']
    two = coupon_offsets(model, 0, 'persuadable', [0, 0])['purchase']
    late = coupon_offsets(model, 0, 'persuadable', [14])['purchase']
    expired = coupon_offsets(model, 0, 'persuadable', [35])['purchase']
    assert one < two < 2 * one and 0 < late < one and expired == 0
    assert coupon_offsets(model, 0, 'persuadable', [0], None)['purchase'] == 2
    persuaded = coupon_action_probabilities(base, model, 0, 'persuadable', [0])
    harmed = coupon_action_probabilities(base, model, 0, 'sleeping_dog', [0])
    assert persuaded[3] > base[3] > harmed[3]


def test_canonical_probabilities_use_the_users_normalized_latent():
    matrix = np.zeros((3, 4))
    matrix[1, 1], matrix[1, 3] = 2., -2.
    model = {
        'canonical_assignment': {
            'types': ['sure_thing', 'persuadable', 'sleeping_dog', 'lost_cause'],
            'matrix': matrix.tolist(),
        }
    }
    _, positive = canonical_type_probabilities(model, 0, np.array([1., 0.]))
    _, negative = canonical_type_probabilities(model, 0, np.array([-1., 0.]))
    assert positive[1] > positive[3]
    assert negative[1] < negative[3]


def test_no_uplift_rollout_connects_coefficients_actions_and_gaps():
    actions = ['view', 'cart', 'remove', 'purchase', 'end']
    features = [
        'intercept', 'log_session_position', 'log_session_elapsed_minutes',
        'log_inter_event_gap_minutes', 'prior_view_count', 'prior_cart_count',
        'prior_remove_count', 'prior_purchase_count', 'previous_view',
        'previous_cart', 'previous_remove', 'previous_purchase',
    ]
    terms = [f'{action}:{feature}' for action in actions[:-1] for feature in features]
    coefficient_model = {
        'weights': [1.], 'standardized_means': [[0.] * len(terms)],
        'standardized_variances': [[0.] * len(terms)],
        'scaler_mean': [0.] * len(terms), 'scaler_scale': [1.] * len(terms),
        'terms': terms, 'dependent_action': 'end',
        'component_persona_composition': [
            {'gmm_component': 0, 'persona': 0, 'size': 10}
        ],
    }
    action_model = {
        'mean': [0.] * 18, 'scale': [1.] * 18, 'actions': actions,
        'features': features + [f's{i}' for i in range(7)],
        'personas': {'0': {'coefficient': np.zeros((5, 19)).tolist()}},
    }
    specification = {
        'memory': 0., 'features': 'internal_state',
        'mean': [0.] * 6, 'scale': [1.] * 6,
        'coefficient': [[0.] * 6 for _ in range(7)],
        'intercept': [30.] + [-30.] * 6,
        'churn_features': 'internal_state',
        'churn_mean': [0.] * 6, 'churn_scale': [1.] * 6,
        'churn_coefficient': [0.] * 6, 'churn_intercept': -30.,
    }
    gap_model = {
        'initial_gap_hours': 1., 'edges_hours': [0, 1, 6, 24, 72, 168, 336, 720],
        'gap_tokens': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'NO_RETURN_30D'],
        'personas': {'0': specification},
        'lifecycle': {'mature_personas': [0], 'home_persona_prior': {'0': 1.}},
    }
    result = simulate_no_uplift_users(
        coefficient_model, action_model, gap_model, users=5, seed=7,
        max_sessions=3, max_events=20,
    )
    assert len(result['sessions']) == 15
    assert result['truncated_user_rate'] == 1
    assert all(row['events'] >= 1 for row in result['sessions'])
    assert {row['home_persona'] for row in result['sessions']} == {0}
