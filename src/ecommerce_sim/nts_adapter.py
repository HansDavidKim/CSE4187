from __future__ import annotations

from pathlib import Path

import numpy as np


def load_sequences(path: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """Load independent customer sequences saved by ``save_amexpert_panel``."""
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["values"], dtype=np.float64)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        features = [str(value) for value in data["features"]]

    if values.ndim != 2:
        raise ValueError("values must have shape [observations, features]")
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError("offsets must contain at least a start and end")
    if offsets[0] != 0 or offsets[-1] != len(values) or np.any(np.diff(offsets) < 0):
        raise ValueError("offsets do not define a valid partition of values")
    if values.shape[1] != len(features):
        raise ValueError("feature count does not match values")

    return [values[start:end] for start, end in zip(offsets[:-1], offsets[1:])], features


def make_lag_windows(
    sequences: list[np.ndarray], max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create [feature, history+target] windows without crossing customer boundaries.

    The returned sequence index can be used for customer-level bootstrap sampling and
    grouped train/validation splits.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    if not sequences:
        raise ValueError("at least one sequence is required")

    feature_count = sequences[0].shape[1]
    windows: list[np.ndarray] = []
    sequence_ids: list[int] = []
    for sequence_id, sequence in enumerate(sequences):
        sequence = np.asarray(sequence, dtype=np.float64)
        if sequence.ndim != 2 or sequence.shape[1] != feature_count:
            raise ValueError("all sequences must be two-dimensional with equal feature counts")
        for stop in range(max_lag, len(sequence)):
            windows.append(sequence[stop - max_lag : stop + 1].T)
            sequence_ids.append(sequence_id)

    if not windows:
        raise ValueError("no sequence is longer than max_lag")
    return np.stack(windows), np.asarray(sequence_ids, dtype=np.int64)
