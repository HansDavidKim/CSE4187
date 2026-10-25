import numpy as np
import pandas as pd

from analysis.persona.behavior import summarize_delays


def test_delay_summary_counts_nonreturners_and_horizon_boundaries():
    df = pd.DataFrame({'gap_hours': [1.0, 24.0, 720.0, np.nan, 721.0],
                       'previous_gap_hours': [3., 4., 5., 6., 7.]})
    out = summarize_delays(df)
    assert out['n'] == 5
    assert out['return_counts'] == [0, 1, 1, 2, 2, 2, 2, 3]
    assert out['returns_30d'] == 3
    assert out['returner_gap_hours_quantiles'][1] == 24
    assert out['paired_next_greater_count'] == 2
    assert summarize_delays(df.iloc[:0]) is None
