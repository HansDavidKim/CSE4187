"""A reversible 28-day lifecycle view of observed session history only."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


INACTIVITY = pd.Timedelta(days=28)
COLUMNS = ["customer_id", "session_id", "as_of_at", "life_cycle", "transition_type", "transition_order",
           "last_session_id", "last_session_started_at", "first_observed_at", "elapsed_since_visit_days",
           "history_left_censored", "home_persona"]


def _utc(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("lifecycle timestamps must be timezone-aware and known")
    return result.tz_convert("UTC")


def build_observed_lifecycle(sessions: pd.DataFrame, metadata: dict,
                             home_personas: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Emit observed visits and elapsed-inactivity transitions in [start, end).

    ENTRY means first *observed* session, never verified signup. Every later
    session is HOME, including returns after EXIT. EXIT is reversible absence
    of a new session start for 28 days; it is not permanent account departure.
    At an exact deadline a timer precedes the same-time returning session.
    """
    start = _utc(metadata.get("observation_start_at", metadata.get("start_at")))
    end = _utc(metadata.get("observation_end_at", metadata.get("end_at")))
    if end <= start:
        raise ValueError("lifecycle observation end must follow start")
    required = {"customer_id", "session_id", "started_at"}
    if not required <= set(sessions):
        raise ValueError(f"lifecycle requires session columns {sorted(required)}")
    visits = sessions[["customer_id", "session_id", "started_at"]].copy()
    for name in ("customer_id", "session_id"):
        if not visits[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{name} must be a nonempty string")
    if visits.session_id.duplicated().any():
        raise ValueError("session_id must be unique")
    if isinstance(visits.started_at.dtype, pd.DatetimeTZDtype):
        visits["started_at"] = visits.started_at.dt.tz_convert("UTC")
        if visits.started_at.isna().any():
            raise ValueError("lifecycle session times must be known")
    else:
        visits["started_at"] = pd.to_datetime(visits.started_at.map(_utc), utc=True)
    visits = visits.loc[visits.started_at.ge(start) & visits.started_at.lt(end)].sort_values(
        ["customer_id", "started_at", "session_id"], kind="stable")
    tags = home_personas or {}
    rows, pending = [], 0
    for customer_id, history in visits.groupby("customer_id", sort=True):
        first = history.started_at.iloc[0]
        previous = None

        def record(at, state, session_id, last_id, last_at, transition):
            rows.append({"customer_id": customer_id, "session_id": session_id, "as_of_at": at,
                         "life_cycle": state, "transition_type": transition,
                         "transition_order": 0 if transition == "inactivity_28d" else 1,
                         "last_session_id": last_id, "last_session_started_at": last_at,
                         "first_observed_at": first,
                         "elapsed_since_visit_days": (at - last_at).total_seconds() / 86400,
                         "history_left_censored": True, "home_persona": tags.get(customer_id)})

        for session in history.itertuples(index=False):
            if previous is not None:
                deadline = previous.started_at + INACTIVITY
                if deadline <= session.started_at:
                    record(deadline, "EXIT", None, previous.session_id, previous.started_at, "inactivity_28d")
            record(session.started_at, "ENTRY" if previous is None else "HOME", session.session_id,
                   session.session_id, session.started_at, "session")
            previous = session
        deadline = previous.started_at + INACTIVITY
        if deadline < end:
            record(deadline, "EXIT", None, previous.session_id, previous.started_at, "inactivity_28d")
        else:
            pending += 1
    result = pd.DataFrame(rows, columns=COLUMNS)
    for name in ("as_of_at", "last_session_started_at", "first_observed_at"):
        result[name] = pd.to_datetime(result[name], utc=True)
    result = result.sort_values(["customer_id", "as_of_at", "transition_order", "session_id"],
                                kind="stable", na_position="first").reset_index(drop=True)
    result.attrs = {
        "definition_id": "observed_session_history_28d_v1", "inactivity_days": 28,
        "observation_start_at": start.isoformat(), "observation_end_at": end.isoformat(),
        "visit_clock": "session started_at; not future activity-window assignment",
        "boundary": "rows in [start,end); at start+28d, EXIT timer precedes a same-time HOME return",
        "entry_meaning": "first observed session; no verified lifetime signup or pre-observation history",
        "exit_meaning": "reversible observed 28-day inactivity, not permanent departure",
        "final_return_gaps_right_censored": int(visits.customer_id.nunique()),
        "exit_deadlines_outside_observation": pending,
        "home_persona_meaning": "supplied latent-profile tag, not an observed or predicted ground-truth class",
        "cart_quantity_after": "omitted; a session-end quantity must not be attached as known at session start",
    }
    return result
