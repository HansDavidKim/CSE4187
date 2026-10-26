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


def test_shared_lifecycle_features_exclude_unobserved_remove_action():
    frame = pd.DataFrame({
        'view': [2, 0], 'cart': [1, 0], 'remove': [20, 10], 'purchase': [0, 1],
        'previous_gap_hours': [np.nan, 24.0],
    })
    values = behavior.shared_lifecycle_window_features(frame)
    assert np.isclose(values['view_share'], .5)
    assert np.isclose(values['cart_share'], .25)
    assert np.isclose(values['purchase_share'], .25)
    assert values['events'] == 4


def test_lifecycle_decisions_respect_30_day_followup_and_censoring():
    sessions = pd.DataFrame({
        'user': [1, 1, 2], 'session': ['a', 'b', 'c'],
        'start': pd.to_datetime(['2020-01-01', '2020-02-10', '2020-03-20'], utc=True),
        'end': pd.to_datetime(['2020-01-01', '2020-02-10', '2020-03-20'], utc=True),
        'next_gap_hours': [960.0, np.nan, np.nan],
        'previous_gap_hours': [np.nan, 960.0, np.nan],
        'duration_hours': [0.0, 0.0, 0.0],
        'view': [1, 1, 1], 'cart': [0, 0, 0], 'remove': [0, 0, 0],
        'purchase': [0, 0, 0],
    })
    result = behavior.lifecycle_decisions(
        sessions, pd.Timestamp('2020-03-31', tz='UTC')
    )
    assert result[['user', 'session']].to_records(index=False).tolist() == [(1, 'a'), (1, 'b')]
    assert result.no_return_30d.tolist() == [1, 1]


def test_lifecycle_intercept_calibration_matches_target_rate():
    linear = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    intercept = behavior._calibrate_intercept(linear, .3)
    probability = 1 / (1 + np.exp(-(linear + intercept)))
    assert np.isclose(probability.mean(), .3)


def test_long_term_stage_row_detects_view_led_churning():
    frame = pd.DataFrame({
        'start': pd.date_range('2020-01-01', periods=12, freq='D', tz='UTC'),
        'view': [1] * 4 + [2] * 4 + [4] * 4,
        'cart': [2] * 4 + [1] * 4 + [0] * 4,
        'remove': [0] * 12,
        'purchase': [1] * 4 + [1] * 4 + [0] * 4,
        'events': [4] * 4 + [4] * 4 + [4] * 4,
        'previous_gap_hours': [1.] * 12,
        'no_return_30d': [0] * 11 + [1],
    })
    row = behavior._long_term_stage_row('cosmetics', 1, frame)
    assert row['terminal_state'] == 'DISSATISFIED_CHURNING'
    assert row['view_shift_10pp']
    assert not row['event_volume_decline']
    assert not row['view_led_churning_proxy']
