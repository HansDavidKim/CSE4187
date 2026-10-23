"""Landmark inactivity labels with explicit administrative right censoring."""
from __future__ import annotations

import pandas as pd


def inactivity_label(landmark, next_start, observation_end, days):
    if days <= 0:
        raise ValueError('days must be positive')
    landmark = pd.to_datetime(landmark, utc=True)
    next_start = pd.to_datetime(next_start, utc=True)
    end = pd.Timestamp(observation_end)
    if end.tzinfo is None:
        raise ValueError('observation end must be timezone-aware')
    cutoff = landmark + pd.Timedelta(days=days)
    if (next_start.notna() & next_start.lt(landmark)).any():
        raise ValueError('next session precedes landmark')
    labels = pd.Series(pd.NA, index=landmark.index, dtype='boolean')
    observed = cutoff.le(end)
    labels.loc[observed] = (next_start.loc[observed].isna() |
                            next_start.loc[observed].gt(cutoff.loc[observed]))
    return labels
