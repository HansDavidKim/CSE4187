import sys

from analysis.uplift import observed


def test_observational_workflows_keep_separate_output_directories():
    paths = {observed.REACT_OUT, observed.PERSIST_OUT, observed.LONG_OUT,
             observed.PERSONA_OUT, observed.PAIRS_OUT, observed.TIANCHI_PROXY_OUT,
             observed.TIANCHI_UPLIFT_OUT, observed.TIANCHI_OVERLAP_OUT, observed.TIANCHI_CUMULATIVE_OUT}
    assert len(paths) == 9


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
