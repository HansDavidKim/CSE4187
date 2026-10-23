import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("reactivation", Path(__file__).resolve().parents[1] / "scripts/coupon_reactivation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_cohort_uses_only_prior_and_quiet_history():
    data = pd.DataFrame({"prior_0": [1, 1, 0, 0], "prior_1": [0, 0, 1, 0],
                         "quiet_0": [0, 1, 0, 0], "quiet_1": [0, 0, 0, 0],
                         "anchor_2": [1, 1, 0, 1], "w1_0": [0, 1, 1, 1]})
    cohort = module.cohort_from_counts(data)
    assert cohort.index.tolist() == [0, 2]
    assert cohort.group.tolist() == ["click_only", "purchase_only"]
    assert cohort.treated.tolist() == [1, 0]
