import pandas as pd

from analysis.uplift import observed as module


def test_cohort_uses_only_prior_and_quiet_history():
    data = pd.DataFrame({"prior_0": [1, 1, 0, 0], "prior_1": [0, 0, 1, 0],
                         "quiet_0": [0, 1, 0, 0], "quiet_1": [0, 0, 0, 0],
                         "anchor_2": [1, 1, 0, 1], "w1_0": [0, 1, 1, 1]})
    cohort = module.cohort_from_counts(data)
    assert cohort.index.tolist() == [0, 2]
    assert cohort.group.tolist() == ["click_only", "purchase_only"]
    assert cohort.treated.tolist() == [1, 0]


def test_tianchi_persona_proxy_rules_are_disjoint():
    frame = pd.DataFrame({"pre_total": [1, 2, 8, 12, 20], "pre_purchase": [0, 0, 0, 1, 3]})
    labels, quantiles = module.tianchi_persona_proxy_labels(frame)
    assert set(labels) <= {"P1_observation_proxy", "P2_deep_consideration_proxy",
                           "P3_cart_revision_proxy", "P4_purchase_path_proxy", "unassigned"}
    assert quantiles["pre_total_q25"] <= quantiles["pre_total_q75"] <= quantiles["pre_total_q95"]


def test_coupon_overlap_bands_cap_at_two_or_more():
    assert module.coupon_overlap_band([0, 1, 2, 7]).tolist() == [0, 1, 2, 2]
