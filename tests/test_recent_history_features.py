import numpy as np
import pandas as pd
import pytest

from ecommerce_sim.recent_history_features import decision_rows, feature_names, state_features


def test_current_action_is_never_in_its_input_and_lags_shift():
    frame = pd.DataFrame({'session': [1, 1, 1, 1, 1], 'at': pd.date_range('2020-01-01', periods=5, freq='s'),
                          'action': [0, 1, 2, 0, 1]})
    rows, _ = decision_rows(frame, 'electronics')
    assert rows[0] == ((), 'session_start', 0)
    assert rows[3][0] == (2, 1, 0)
    assert rows[4][0] == (0, 2, 1, 0)
    v = state_features(rows[3][0], rows[3][1], 2)
    names = feature_names(2)
    assert v[names.index('lag0_purchase')] == 1
    assert v[names.index('lag1_cart')] == 1
    assert v[names.index('lag0_purchase*lag1_cart')] == 1


def test_boundaries_remove_and_timestamps():
    frame = pd.DataFrame({'session': [1, 1, 1, 1, 1, 2],
                          'at': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03',
                                               '2020-01-03', '2020-01-04', '2020-01-05']),
                          'action': [0, 2, 0, 1, 3, 0]})
    rows, audit = decision_rows(frame, 'cosmetics')
    assert rows == [((), 'session_start', 0), ((), 'unknown', 2), ((), 'session_start', 0)]
    assert audit == {'tied_events_skipped': 2, 'unsupported_events': 1}
    assert state_features([], 'unknown', 3)[2] == 1
    with pytest.raises(ValueError):
        feature_names(4)
