"""Private observed-session memory; never an additional reward predictor."""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

LIFECYCLE_MEMORY = "first_two_observed_sessions"
SESSION_NUMBER = "observed_session_number"


def validate_lifecycle_memory(value):
    if value is not None and value != LIFECYCLE_MEMORY:
        raise ValueError("unknown private lifecycle memory")
    return value


def phase_from_session_number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("private lifecycle memory requires a positive observed_session_number")
    return int(value > 2)


def phase_ids(records):
    if SESSION_NUMBER not in records:
        raise ValueError("private lifecycle memory requires original-prefix observed_session_number")
    values = np.asarray(records[SESSION_NUMBER])
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 1).any():
        raise ValueError("observed_session_number must retain positive integer prefix ordinals")
    return (values > 2).astype(np.int8)


def attach_session_numbers(data, source_rows):
    """Count all original observed sessions before any validity/fit selection."""
    required = {"user_id", "session", "at", "source_event_id"}
    if source_rows.empty or not required <= set(source_rows):
        raise ValueError("private session memory requires the complete original source prefix")
    if (not isinstance(source_rows["at"].dtype, pd.DatetimeTZDtype)
            or source_rows[["user_id", "session", "at", "source_event_id"]].isna().any().any()):
        raise ValueError("private source session keys/clocks must be observed")
    ordered = source_rows.sort_values(["user_id", "at", "source_event_id"], kind="stable")
    sessions = ordered.drop_duplicates(["user_id", "session"], keep="first").copy()
    sessions["user_id"] = sessions.user_id.astype(str)
    sessions[SESSION_NUMBER] = sessions.groupby("user_id", sort=False).cumcount() + 1
    lookup = sessions.set_index(["user_id", "session"])[SESSION_NUMBER]
    for name in ("records", "sessions"):
        frame = data[name]
        keys = pd.MultiIndex.from_arrays([frame.user_id.astype(str), frame.session])
        numbers = lookup.reindex(keys)
        if numbers.isna().any():
            raise ValueError("fitting rows are not aligned to the original source sessions")
        frame[SESSION_NUMBER] = numbers.to_numpy(dtype=np.int64)
    data["lifecycle_memory"] = LIFECYCLE_MEMORY
    data["audit"]["private_lifecycle_memory"] = {
        "rule": LIFECYCLE_MEMORY, "phase_0": "first two observed sessions", "phase_1": "third and later observed sessions",
        "session_order": "all original-prefix sessions ordered by source time and original event order before filtering",
        "invalid_and_censored_sessions_count": True, "not_a_reward_predictor": True,
        "inactivity_and_t0_reset": False,
    }
    return data
