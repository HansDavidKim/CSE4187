import sys

from analysis import eda


def test_eda_dispatches_requested_task(monkeypatch):
    calls = []
    monkeypatch.setattr(eda, 'evaluate', calls.append)
    monkeypatch.setattr(sys, 'argv', ['eda', 'session-return', 'cosmetics'])
    eda.main()
    assert calls == ['cosmetics']


def test_eda_dispatches_audit_options(monkeypatch):
    calls = []
    monkeypatch.setattr(eda, 'audit_main', calls.append)
    monkeypatch.setattr(sys, 'argv', ['eda', 'audit', '--amexpert', '/data'])
    eda.main()
    assert calls == [['--amexpert', '/data']]
