import numpy as np
import pandas as pd

from analysis.irl.persona import (
    ACTIONS,
    FEATURES,
    _gap_choice_labels,
    _gap_states,
    action_probabilities,
    brief_browser_subtypes,
    fit_coefficient_gmm,
    make_decisions,
    maxent_objective,
    quadratic_design,
    select_representatives,
)


def test_maxent_gradient_matches_finite_difference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(12, 4))
    y = rng.integers(0, len(ACTIONS), size=12)
    flat = rng.normal(scale=.1, size=(len(ACTIONS) - 1) * x.shape[1])
    _, gradient = maxent_objective(flat, x, y, np.zeros((len(ACTIONS), 4)), .03)
    epsilon = 1e-6
    numeric = np.empty_like(flat)
    for i in range(len(flat)):
        plus, minus = flat.copy(), flat.copy()
        plus[i] += epsilon
        minus[i] -= epsilon
        numeric[i] = (
            maxent_objective(plus, x, y, np.zeros((len(ACTIONS), 4)), .03)[0]
            - maxent_objective(minus, x, y, np.zeros((len(ACTIONS), 4)), .03)[0]
        ) / (2 * epsilon)
    assert np.allclose(gradient, numeric, atol=1e-6)


def test_reference_reward_and_representative_selection():
    probabilities = action_probabilities(np.ones((2, 2)), np.zeros((len(ACTIONS), 2)))
    assert np.allclose(probabilities, .2)
    frame = pd.DataFrame({
        "user_id": range(8), "cluster": [0] * 4 + [1] * 4, "sessions": [2] * 8,
        "z_0": [0, .1, 1, 2, 10, 10.2, 12, 13], "z_1": [0] * 8,
    })
    selected = select_representatives(frame, np.array([[0, 0], [10, 0]]), per_persona=2)
    assert selected.user_id.tolist() == [0, 1, 4, 5]


def test_decisions_use_past_state_and_add_observable_end():
    events = pd.DataFrame({
        "user": [1, 1, 1, 2], "session": ["a", "a", "b", "c"],
        "at": pd.to_datetime(["2020-01-01", "2020-01-01 00:01",
                              "2020-01-01 00:10", "2020-01-02"],
                             format="mixed", utc=True),
        "action": [0, 1, 0, 0],
    })
    assignments = pd.DataFrame({"user_id": [1, 2], "cluster": [0, 1]})
    decisions = make_decisions(events, assignments)
    first, second = decisions.loc[decisions.user_id.eq(1)].iloc[:2].to_dict("records")
    assert first["prior_view_count"] == 0
    assert second["prior_view_count"] == np.log(2)
    # No inferred END may be inserted after session a because b starts before
    # the 30-minute inactivity boundary.
    assert decisions.loc[decisions.user_id.eq(1), "action"].tolist() == [0, 1, 0, 4]
    end = decisions.loc[(decisions.user_id.eq(1)) & (decisions.action.eq(4))].iloc[0]
    assert end.log_inter_event_gap_minutes == 0


def test_quadratic_design_has_squares_and_pairwise_terms_only():
    x = np.arange(3 * len(FEATURES), dtype=float).reshape(3, len(FEATURES))
    x[:, 0] = 1.0
    expanded, names = quadratic_design(x, np.array([0, 1]))
    expected = len(FEATURES) + len(FEATURES[1:]) + len(FEATURES[1:]) * (len(FEATURES[1:]) - 1) // 2
    assert expanded.shape == (3, expected)
    assert names[:len(FEATURES)] == FEATURES
    assert "square:log_session_position" in names
    assert "interaction:log_session_position*log_session_elapsed_minutes" in names
    assert not any(name.count("*") > 1 for name in names)


def test_gap_states_can_use_personal_oracle_initial_magnitude():
    frame = pd.DataFrame({
        "user": [1, 1, 2], "view": [1, 1, 1], "cart": [0, 0, 0],
        "remove": [0, 0, 0], "purchase": [0, 0, 0],
        "duration_hours": [0, 0, 0], "events": [1, 1, 1],
        "previous_gap_hours": [np.nan, 12.0, np.nan],
    })
    states = _gap_states(frame, 48.0, {1: 24.0})
    assert np.isclose(states[0, -3], np.log1p(24))
    assert np.isclose(states[1, -3], np.log1p(12))
    assert np.isclose(states[2, -3], np.log1p(48))
    assert np.allclose(states[:2, -2], np.log1p(24))
    assert np.isclose(states[2, -2], np.log1p(48))
    assert np.all(states[[0, 2], -1] == 0)


def test_gap_tokens_require_observed_return_or_full_30_day_followup():
    end = pd.Timestamp('2020-02-01', tz='UTC')
    frame = pd.DataFrame({
        'end': [end, end, pd.Timestamp('2020-02-29', tz='UTC')],
        'next_gap_hours': [2.0, np.nan, np.nan],
    })
    labels = _gap_choice_labels(frame, pd.Timestamp('2020-03-02', tz='UTC'))
    assert labels.tolist() == [1, 7, -1]


def test_brief_browser_subtypes_separate_birth_and_lifecycle_roles():
    rows = [
        (1, 'a', '2020-01-01', 1, 0),
        (2, 'a', '2020-02-20', 1, 0),
        (3, 'a', '2020-01-01', 2, 0), (3, 'b', '2020-01-05', 1, 0),
        (4, 'a', '2020-01-01', 1, 0), (4, 'b', '2020-02-20', 1, 1),
    ]
    frame = pd.DataFrame(rows, columns=['user', 'session', 'start', 'unique_products', 'purchase'])
    frame['start'] = pd.to_datetime(frame.start, utc=True)
    frame['end'] = frame.start
    frame['persona'] = 1
    frame['cart'] = frame['remove'] = 0
    frame['events'] = 1
    frame['previous_gap_hours'] = frame.groupby('user').start.diff().dt.total_seconds() / 3600
    users, _ = brief_browser_subtypes(frame, pd.Timestamp('2020-03-01', tz='UTC'))
    states = users.set_index('user').subtype.to_dict()
    assert states == {1: 'AMBIGUOUS_SINGLE_VIEW_EXIT', 2: 'NEW',
                      3: 'DISSATISFIED_CHURNING', 4: 'REGULAR'}


def test_coefficient_gmm_persists_identifiable_distribution(tmp_path):
    rows = []
    for user in range(20):
        for action_index, action in enumerate(ACTIONS):
            for feature_index, feature in enumerate(FEATURES):
                rows.append({
                    'user_id': user, 'persona': (0, 2, 3)[user % 3],
                    'action': action, 'feature': feature,
                    'coefficient': (user % 2) * 2 + action_index * .1 + feature_index * .01,
                })
    result = fit_coefficient_gmm(pd.DataFrame(rows), tmp_path, seed=3, max_components=3)
    assert 1 <= result['selected_components'] <= 3
    assert np.isclose(sum(result['weights']), 1)
    assert len(result['terms']) == (len(ACTIONS) - 1) * len(FEATURES)
    assert (tmp_path / 'coefficient_gmm_parameters.csv').exists()


