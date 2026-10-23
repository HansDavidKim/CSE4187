import pytest

from ecommerce_sim.persona_evolution import session_tokens, summarize_tokens


def test_session_windows_do_not_share_events_and_truncate():
    sessions = [[0, 1], [2], [0], [0, 2]]
    assert session_tokens(sessions[:2]) == [0, 1, 3, 2, 3]
    assert session_tokens(sessions[2:]) == [0, 3, 0, 2, 3]
    assert session_tokens(sessions, max_tokens=3) == [0, 1, 3]
    assert summarize_tokens([0, 1, 3, 2, 3])['events'] == 3


def test_invalid_sessions_are_rejected():
    with pytest.raises(ValueError):
        session_tokens([[4]])
    with pytest.raises(ValueError):
        session_tokens([[]])
