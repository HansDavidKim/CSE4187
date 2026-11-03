"""Common event-only trajectory features for three commerce event datasets."""
from __future__ import annotations

import numpy as np

COMMON_ACTIONS = ('view', 'cart', 'purchase')
COEFFICIENT_NAMES = tuple(f'{s}_to_{a}' for s in COMMON_ACTIONS for a in COMMON_ACTIONS)[1:]


def trajectory_windows(tokens: list[int], actions: list[str], horizon: int = 3):
    """Create disjoint windows; unmodeled events and session ends break sequences."""
    if horizon < 1 or not {'view', 'cart', 'purchase'}.issubset(actions):
        raise ValueError('positive horizon and common actions required')
    mapping = {actions.index(name): i for i, name in enumerate(COMMON_ACTIONS)}
    states, choices = [], []
    segment = []
    for token in [*tokens, -1]:
        if token in mapping:
            segment.append(mapping[token])
            continue
        for start in range(0, len(segment) - horizon, horizon + 1):
            window = segment[start:start + horizon + 1]
            states.append(window[:-1])
            choices.append(window[1:])
        segment = []
    return (np.asarray(states, dtype=np.int64).reshape(-1, horizon),
            np.asarray(choices, dtype=np.int64).reshape(-1, horizon))
