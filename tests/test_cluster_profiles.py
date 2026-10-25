import numpy as np
from analysis.persona.cluster import _fixed_mahalanobis_assignment, profile


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
