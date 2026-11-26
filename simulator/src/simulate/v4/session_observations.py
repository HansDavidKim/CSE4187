"""Completion availability for recorded visits, without changing visit counts."""
from __future__ import annotations

import hashlib
import json

import pandas as pd


def completion_digest(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completion_exclusions(metadata, session_ids=None):
    contract = metadata.get("session_completion")
    if contract is None:
        return None
    if (not isinstance(contract, dict) or set(contract) != {"clock", "censored_session_ids"}
            or contract["clock"] != "ended_at_plus_30min_v1"):
        raise ValueError("unsupported public session completion contract")
    excluded = contract["censored_session_ids"]
    if (not isinstance(excluded, list) or any(not isinstance(value, str) or not value for value in excluded)
            or len(set(excluded)) != len(excluded)
            or session_ids is not None and not set(excluded) <= set(session_ids)):
        raise ValueError("public session completion requires unique known censored session IDs")
    identity = metadata.get("dataset_id")
    if not isinstance(identity, str) or not identity.endswith("-completion-" + completion_digest(contract)):
        raise ValueError("public session completion differs from its dataset identity")
    return set(excluded)


def session_availability(tables):
    sessions = tables["sessions"].copy()
    for name in ("started_at", "ended_at"):
        sessions[name] = pd.to_datetime(sessions[name], utc=True, errors="raise")
    excluded = completion_exclusions(tables.get("_metadata", {}))
    if excluded is not None:
        sessions["completion_available_at"] = (sessions.ended_at + pd.Timedelta(minutes=30)).where(
            ~sessions.session_id.isin(excluded))
        return sessions
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
