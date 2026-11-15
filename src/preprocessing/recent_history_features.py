"""Recent observed-action state features, before the target decision."""
from itertools import combinations

import numpy as np

ACTIONS = ('view', 'cart', 'purchase')


def feature_names(max_lag):
    if max_lag not in range(4):
        raise ValueError('max_lag must be in 0..3')
    main = [f'lag{lag}_{action}' for lag in range(max_lag + 1) for action in ACTIONS]
    interactions = [f'lag{i}_{a}*lag{j}_{b}' for i, j in combinations(range(max_lag + 1), 2)
                    for a in ACTIONS for b in ACTIONS]
    return ('intercept', 'boundary_session_start', 'boundary_unknown', *main, *interactions)


def state_features(recent, boundary, max_lag):
    """recent is newest-first; boundary is session_start, unknown, or complete."""
    if boundary not in ('session_start', 'unknown', 'complete'):
        raise ValueError('invalid boundary type')
    if len(recent) > 4 or any(a not in range(3) for a in recent):
        raise ValueError('recent actions must be common action IDs')
    names = feature_names(max_lag)
    values = np.zeros(len(names), dtype=np.float64)
    values[0] = 1
    incomplete = len(recent) < max_lag + 1
    values[1] = incomplete and boundary == 'session_start'
    values[2] = incomplete and boundary == 'unknown'
    pos = 3
    for lag in range(max_lag + 1):
        if lag < len(recent):
            values[pos + 3 * lag + recent[lag]] = 1
    pos += 3 * (max_lag + 1)
    for i, j in combinations(range(max_lag + 1), 2):
        if j < len(recent):
            values[pos + 3 * recent[i] + recent[j]] = 1
        pos += 9
    return values


def decision_rows(frame, source):
    """One record per unambiguous common-action event; never cross time ties."""
    action_map = {0: 0, 1: 1, 3: 2} if source == 'cosmetics' else {0: 0, 1: 1, 2: 2}
    rows, tied_events, unsupported = [], 0, 0
    records = list(frame[['session', 'at', 'action']].itertuples(index=False, name=None))
    recent, boundary, previous_session = [], 'session_start', None
    i = 0
    while i < len(records):
        session, timestamp, raw = records[i]
        if session != previous_session:
            recent, boundary, previous_session = [], 'session_start', session
        j = i + 1
        while j < len(records) and records[j][:2] == (session, timestamp):
            j += 1
        if j - i > 1:
            tied_events += j - i
            recent, boundary = [], 'unknown'
        elif raw not in action_map:
            unsupported += 1
            recent, boundary = [], 'unknown'
        else:
            action = action_map[raw]
            rows.append((tuple(recent), boundary, action))
            recent = [action, *recent][:4]
            if len(recent) == 4:
                boundary = 'complete'
        i = j
    return rows, {'tied_events_skipped': tied_events, 'unsupported_events': unsupported}
