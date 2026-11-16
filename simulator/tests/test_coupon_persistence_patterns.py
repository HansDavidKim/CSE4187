import numpy as np

from analysis.uplift import observed as module


def test_event_week_statistics_include_gaps_and_boundary():
    bits = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0]], dtype=bool)
    x = module.summarize_persistence(bits)
    assert x["active_weeks_distribution"][:3] == [1, 1, 1]
    assert x["first_recorded_week_distribution_0_none"][:3] == [1, 1, 1]
    assert x["last_recorded_week_distribution_0_none"][6] == 1
    assert x["week1_and_week6_users"] == 1
