import numpy as np
import pandas as pd

from analysis.persona import behavior, cluster


def test_persona_workflows_keep_distinct_output_locations():
    assert cluster.CLUSTER_OUT != cluster.SWEEP_OUT
    assert cluster.SWEEP_OUT == cluster.STABILITY_OUT
    assert len({behavior.WINDOW_OUT, behavior.DYNAMICS_OUT,
                behavior.DELAY_OUT, behavior.HETERO_OUT}) == 4
    assert behavior.DELAY_FIG != behavior.HETERO_FIG


def test_persona_workflows_are_available():
    for function in (cluster.run_clustering, cluster.run_sweep, cluster.run_stability,
                     behavior.run_window_comparison, behavior.run_dynamics,
                     behavior.run_delays, behavior.run_heterogeneity,
                     behavior.plot_heterogeneity, behavior.run_lifecycle):
        assert callable(function)


def test_lifecycle_features_keep_action_shares_and_gap_magnitude():
    frame = pd.DataFrame({
        'view': [2, 0], 'cart': [1, 0], 'remove': [0, 0], 'purchase': [0, 1],
        'previous_gap_hours': [np.nan, 24.0],
    })
    values = behavior.lifecycle_window_features(frame)
    assert np.allclose(values[:4], [.5, .25, 0, .25])
    assert np.isclose(np.expm1(values[-1]), 24)
