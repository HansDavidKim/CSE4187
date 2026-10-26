import sys

import numpy as np
import pandas as pd

from analysis.uplift import observed


def test_observational_workflows_keep_separate_output_directories():
    paths = {observed.REACT_OUT, observed.PERSIST_OUT, observed.LONG_OUT,
             observed.PERSONA_OUT, observed.PAIRS_OUT, observed.TIANCHI_PROXY_OUT,
             observed.TIANCHI_UPLIFT_OUT, observed.TIANCHI_OVERLAP_OUT,
             observed.TIANCHI_CUMULATIVE_OUT, observed.TIANCHI_FIRST_RECEIPT_OUT,
             observed.TIANCHI_PERSONA_COUPON_OUT, observed.TIANCHI_LONG_TERM_OUT,
             observed.TIANCHI_CANONICAL_PROXY_OUT}
    assert len(paths) == 13


def test_observational_cli_dispatches_selected_workflow(monkeypatch):
    calls = []
    monkeypatch.setattr(observed, "run_reactivation", lambda: calls.append("reactivation"))
    monkeypatch.setattr(sys, "argv", ["observed", "reactivation"])
    observed.main()
    assert calls == ["reactivation"]


def test_observational_cli_dispatches_persona_uplift(monkeypatch):
    calls = []
    monkeypatch.setattr(observed, "run_tianchi_persona_uplift", lambda: calls.append("persona-uplift"))
    monkeypatch.setattr(sys, "argv", ["observed", "persona-uplift"])
    observed.main()
    assert calls == ["persona-uplift"]


def test_observational_cli_dispatches_single_receipt_duration(monkeypatch):
    calls = []
    monkeypatch.setattr(observed, "run_tianchi_single_receipt_duration", lambda: calls.append("single-receipt-duration"))
    monkeypatch.setattr(sys, "argv", ["observed", "single-receipt-duration"])
    observed.main()
    assert calls == ["single-receipt-duration"]


def test_observational_cli_dispatches_coupon_overlap(monkeypatch):
    calls = []
    monkeypatch.setattr(observed, "run_tianchi_coupon_overlap", lambda: calls.append("coupon-overlap"))
    monkeypatch.setattr(sys, "argv", ["observed", "coupon-overlap"])
    observed.main()
    assert calls == ["coupon-overlap"]


def test_observational_cli_dispatches_cumulative_coupon_history(monkeypatch):
    calls = []
    monkeypatch.setattr(observed, "run_tianchi_cumulative_coupon_history", lambda: calls.append("cumulative-coupons"))
    monkeypatch.setattr(sys, "argv", ["observed", "cumulative-coupons"])
    observed.main()
    assert calls == ["cumulative-coupons"]


def test_observational_cli_dispatches_first_receipt_alignment(monkeypatch):
    calls = []
    monkeypatch.setattr(
        observed, "run_tianchi_first_receipt_alignment",
        lambda: calls.append("first-receipt-alignment"),
    )
    monkeypatch.setattr(sys, "argv", ["observed", "first-receipt-alignment"])
    observed.main()
    assert calls == ["first-receipt-alignment"]


def test_observational_cli_dispatches_persona_coupon_matching(monkeypatch):
    calls = []
    monkeypatch.setattr(
        observed, "run_tianchi_persona_coupon_matching",
        lambda: calls.append("persona-coupon-matching"),
    )
    monkeypatch.setattr(sys, "argv", ["observed", "persona-coupon-matching"])
    observed.main()
    assert calls == ["persona-coupon-matching"]


def test_observational_cli_dispatches_tianchi_long_term_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(
        observed, "run_tianchi_long_term_lifecycle",
        lambda: calls.append("tianchi-long-term-lifecycle"),
    )
    monkeypatch.setattr(sys, "argv", ["observed", "tianchi-long-term-lifecycle"])
    observed.main()
    assert calls == ["tianchi-long-term-lifecycle"]


def test_observational_cli_dispatches_canonical_proxies(monkeypatch):
    calls = []
    monkeypatch.setattr(
        observed, "run_tianchi_canonical_proxy_estimation",
        lambda: calls.append("canonical-proxies"),
    )
    monkeypatch.setattr(sys, "argv", ["observed", "canonical-proxies"])
    observed.main()
    assert calls == ["canonical-proxies"]


def test_canonical_proxy_bootstrap_preserves_simple_pair_marginals():
    pairs = pd.DataFrame({
        "group": ["S", "S", "S", "S"],
        "pair": [0, 1, 2, 3],
        "control": [0, 0, 1, 1],
        "treated": [0, 1, 1, 1],
    })
    row = observed._canonical_proxy_row("P", {"S": pairs}, {"S": 1.}, draws=200, seed=1)
    assert row["p0"] == .5
    assert row["p1"] == .75
    assert np.isclose(sum(row[name] for name in (
        "sure_thing", "persuadable", "sleeping_dog", "lost_cause"
    )), 1.)


def test_propensity_pairs_are_exact_and_without_replacement():
    frame = pd.DataFrame({
        "treated": [1, 1, 0, 0, 0, 0],
        "pre_purchase_any": [0, 1, 0, 1, 0, 1],
        "x": [0.0, 1.0, 0.01, 1.01, -0.1, 0.9],
    })
    pairs, _ = observed._propensity_pairs(frame, ["x"], caliper_scale=10)
    assert len(pairs) == 2
    assert len(set(pairs[:, 1])) == 2
    for treated, control in pairs:
        assert frame.loc[treated, "pre_purchase_any"] == frame.loc[control, "pre_purchase_any"]
