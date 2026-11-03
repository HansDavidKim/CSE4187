import sys

from analysis.irl import history


def test_l1_cli_preserves_explicit_penalty_options(monkeypatch):
    calls = []
    monkeypatch.setattr(history, 'run_l1', lambda source, l1_only, lags:
                        calls.append((source, l1_only, lags)))
    monkeypatch.setattr(sys, 'argv',
                        ['history', 'l1', 'electronics', '--l1-only', '--lags'])
    history.main()
    assert calls == [('electronics', True, True)]
