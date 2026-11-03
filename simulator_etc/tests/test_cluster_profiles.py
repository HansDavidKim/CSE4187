import numpy as np
import pandas as pd
from analysis.persona.cluster import (
    _fixed_mahalanobis_assignment,
    _nearest_profile_mapping,
    _retail_behavior_features,
    _shared_behavior_features,
    behavioral_profile_similarity,
    profile,
)


def test_profile_excludes_session_boundary_and_handles_empty_cluster():
    groups = profile([[0, 0, 3, 1], [0, 2, 3]], ['view', 'cart', 'purchase'], np.array([0, 0]), 2)
    assert groups[0]['users'] == 2
    assert groups[0]['event_share']['view'] == 3 / 5
    assert groups[0]['users_with_action']['purchase'] == .5
    assert groups[1]['users'] == 0 and groups[1]['event_share']['view'] is None


def test_fixed_mahalanobis_assignment_recovers_separated_anchor_modes():
    z = np.array([[-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0],
                  [2.0, -1.0], [2.0, 0.0], [2.0, 1.0], [0.0, 0.0]])
    core_mask = np.array([True, True, True, True, True, True, False])
    labels = np.array(['left', 'left', 'left', 'right', 'right', 'right'])
    assigned, distance, _, _ = _fixed_mahalanobis_assignment(z, labels, core_mask, ['left', 'right'])
    assert assigned.tolist() == [0, 0, 0, 1, 1, 1, 0]
    assert distance.shape == (7, 2)


def test_behavioral_similarity_separates_shape_from_coverage():
    reference = {
        'view_share': .8, 'cart_share': .1, 'remove_share': .02,
        'purchase_share': .08, 'buyer_share': .2,
        'log_activity_depth': 1.5, 'log_sessions': 1.,
    }
    scales = {name: 1. for name in reference}
    identical, full_coverage, distance = behavioral_profile_similarity(reference, reference, scales)
    partial, partial_coverage, _ = behavioral_profile_similarity(
        reference, {'view_share': .8, 'purchase_share': .08}, scales
    )
    assert identical == partial == 1.
    assert distance == 0.
    assert full_coverage == 1.
    assert partial_coverage == 2 / 7


def test_shared_behavior_features_preserve_common_sequence_direction():
    events = pd.DataFrame({
        'user': [1, 1, 1, 2],
        'at': pd.to_datetime(['2026-01-01', '2026-01-01', '2026-01-03', '2026-01-02'], utc=True),
        'action': ['explore', 'explore', 'purchase', 'purchase'],
    })
    membership = pd.DataFrame({'user_id': [1, 2], 'cluster': [0, 1]})

    features = _shared_behavior_features(events, membership).set_index('user_id')

    assert features.loc[1, 'purchase_share'] == 1 / 3
    assert features.loc[1, 'purchase_day_share'] == .5
    assert features.loc[1, 'explore_to_purchase'] == 1
    assert features.loc[1, 'purchase_to_explore'] == 0
    assert features.loc[1, 'has_transition'] == 1
    assert features.loc[2, 'has_transition'] == 0


def test_nearest_profile_mapping_allows_many_to_one_targets():
    nearest, margin = _nearest_profile_mapping(np.array([[.9, .1], [.8, .2], [.3, .7]]))

    assert nearest.tolist() == [0, 0, 1]
    assert np.allclose(margin, [.8, .6, .4])


def test_retail_features_use_only_shared_actions_and_within_session_transitions():
    events = pd.DataFrame({
        'user_id': [1, 1, 1, 2],
        'session': ['a', 'a', 'a', 'b'],
        'at': pd.to_datetime([
            '2026-01-01 00:00', '2026-01-01 00:01',
            '2026-01-01 00:02', '2026-01-02 00:00',
        ], utc=True),
        'action': ['view', 'cart', 'purchase', 'view'],
    })
    membership = pd.DataFrame({'user_id': [1, 2], 'cluster': [0, 1]})

    features = _retail_behavior_features(events, membership).set_index('user_id')

    assert features.loc[1, 'cart_share'] == 1 / 3
    assert features.loc[1, 'purchase_share'] == 1 / 3
    assert features.loc[1, 'view_to_cart'] == 1
    assert features.loc[1, 'cart_to_purchase'] == 1
    assert features.loc[2, 'view_to_cart'] == 0
