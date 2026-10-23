import numpy as np
from scripts.compare_masked_gru_clusters import profile


def test_profile_excludes_session_boundary_and_handles_empty_cluster():
    groups = profile([[0, 0, 3, 1], [0, 2, 3]], ['view', 'cart', 'purchase'], np.array([0, 0]), 2)
    assert groups[0]['users'] == 2
    assert groups[0]['event_share']['view'] == 3 / 5
    assert groups[0]['users_with_action']['purchase'] == .5
    assert groups[1]['users'] == 0 and groups[1]['event_share']['view'] is None
