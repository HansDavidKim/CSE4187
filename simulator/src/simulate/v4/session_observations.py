"""Completion availability for recorded visits, without changing visit counts."""
from __future__ import annotations

import pandas as pd


def session_availability(tables):
    sessions = tables["sessions"].copy()
    for name in ("started_at", "ended_at"):
        sessions[name] = pd.to_datetime(sessions[name], utc=True, errors="raise")
    diagnostics = tables.get("session_diagnostics")
    if diagnostics is None:
        if "completion_available_at" not in sessions:
            sessions["completion_available_at"] = sessions.ended_at
        else:
            sessions["completion_available_at"] = pd.to_datetime(
                sessions.completion_available_at, utc=True, errors="raise")
        return sessions
    if sessions.empty:
        sessions["completion_available_at"] = pd.Series(dtype="datetime64[ns, UTC]")
        return sessions
    from .validation import _session_completion
    upper = max(sessions.started_at.max(), sessions.ended_at.max()) + pd.Timedelta(nanoseconds=1)
    visits, _ = _session_completion(sessions, sessions.started_at.min(), upper, diagnostics)
    available = (visits.ended_at + pd.Timedelta(minutes=30)).where(visits._complete)
    by_session = pd.Series(available.array, index=visits.session_id)
    sessions["completion_available_at"] = sessions.session_id.map(by_session)
    return sessions
