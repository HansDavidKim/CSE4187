"""Past-only session features and discrete return hazard training examples."""
import numpy as np
import pandas as pd

TIMEOUT = pd.Timedelta(minutes=30)
EDGES = np.array([0, 1, 6, 24, 72, 168, 336, 720, 1440], dtype=float)
FEATURES = ['view', 'cart', 'purchase', 'remove', 'events', 'duration_hours',
            'previous_sessions', 'previous_gap_hours', 'has_previous', 'hour_sin', 'hour_cos']


def sessions(events, observation_end):
    e = events.sort_values(['user', 'at'], kind='stable').copy()
    delta = e.groupby('user', sort=False)['at'].diff()
    e['sid'] = delta.isna() | delta.gt(TIMEOUT)
    e['sid'] = e.groupby('user', sort=False)['sid'].cumsum()
    for action in ('view', 'cart', 'purchase', 'remove'):
        e[action] = e['action'].eq(action).astype('int8')
    e['events'] = 1
    grouped = e.groupby(['user', 'sid'], sort=False)
    s = grouped.agg(start=('at', 'min'), last=('at', 'max'),
                    **{a: (a, 'sum') for a in ('view', 'cart', 'purchase', 'remove', 'events')}).reset_index()
    s['end'] = s['last'] + TIMEOUT
    s['next_start'] = s.groupby('user', sort=False)['start'].shift(-1)
    # Session ends are observed only after the inactivity timeout has elapsed.
    s = s[s['end'] <= observation_end].copy()
    g = s.groupby('user', sort=False)
    s['previous_end'] = g['end'].shift(1)
    s['has_previous'] = s['previous_end'].notna().astype(int)
    s['previous_gap_hours'] = ((s['start'] - s['previous_end']).dt.total_seconds() / 3600).fillna(0).clip(0, 720) / 720
    s['previous_sessions'] = g.cumcount().clip(0, 20) / 20
    s['duration_hours'] = ((s['last'] - s['start']).dt.total_seconds() / 3600).clip(0, 24) / 24
    for col in ('view', 'cart', 'purchase', 'remove', 'events'):
        s[col] = s[col].clip(0, 20) / 20
    hour = s['end'].dt.hour + s['end'].dt.minute / 60
    s['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    s['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return s


def durations(s, cutoff):
    available = (cutoff - s['end']).dt.total_seconds().to_numpy() / 3600
    next_hours = (s['next_start'] - s['end']).dt.total_seconds().to_numpy() / 3600
    event = np.isfinite(next_hours) & (next_hours <= available)
    return np.where(event, next_hours, available), event, available


def expand_hazards(s, cutoff):
    elapsed, event, available = durations(s, cutoff)
    arrays, labels = [], []
    x = s[FEATURES].to_numpy(dtype=float)
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        # Require full administrative follow-up for the entire interval for
        # both outcomes; otherwise early observed returns are overrepresented.
        use = (available >= hi) & (elapsed >= lo)
        occurred = event & (elapsed < hi)
        if not use.any():
            continue
        bin_columns = np.zeros((use.sum(), len(EDGES) - 1))
        bin_columns[:, i] = 1
        arrays.append(np.column_stack((bin_columns, x[use])))
        labels.append(occurred[use].astype(int))
    return np.concatenate(arrays), np.concatenate(labels)


def horizon_prob(model, x, horizon, baseline=False):
    survival = np.ones(len(x))
    for i, hi in enumerate(EDGES[1:]):
        if hi > horizon:
            break
        bins = np.zeros((len(x), len(EDGES) - 1))
        bins[:, i] = 1
        inp = bins if baseline else np.column_stack((bins, x))
        survival *= 1 - model.predict_proba(inp)[:, 1]
    return 1 - survival
