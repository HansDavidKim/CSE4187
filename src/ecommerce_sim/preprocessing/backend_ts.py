"""Leakage-checked Backend-TS history aggregation."""
import numpy as np
import pandas as pd

METHODS = ('LOGIN', 'GET', 'POST', 'PUT', 'DELETE')


def hourly_history(events: pd.DataFrame, index: pd.Timestamp) -> np.ndarray:
    if not {'timestamp', 'method'}.issubset(events):
        raise ValueError('missing timestamp or method')
    ts = pd.to_datetime(events.timestamp, errors='raise')
    if ts.isna().any() or ((ts < index - pd.Timedelta(days=7)) | (ts >= index)).any():
        raise ValueError('event outside seven-day pre-index history')
    if not events.method.isin(METHODS).all():
        raise ValueError('unknown event method')
    hours = ((ts - (index - pd.Timedelta(days=7))) // pd.Timedelta(hours=1)).to_numpy(dtype=int)
    methods = pd.Categorical(events.method, categories=METHODS).codes
    result = np.zeros((168, len(METHODS)), dtype=np.float32)
    np.add.at(result, (hours, methods), 1)
    return np.log1p(result)
