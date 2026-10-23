"""Leakage-safe episode grouping and fixed-grid feature assembly."""
import numpy as np
import pandas as pd


def disjoint_split(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {'user_id', 'episode', 'split', 'sequence_index', 'treatment', 'login_within_3h', 'index_time'}
    if not required.issubset(labels) or labels[list(required)].isna().any().any():
        raise ValueError('missing required episode metadata')
    if not labels.split.isin(['train', 'validation', 'test']).all():
        raise ValueError('unexpected split')
    if not labels.treatment.isin([0, 1]).all() or not labels.login_within_3h.isin([0, 1]).all():
        raise ValueError('T and Y must be binary')
    if labels.duplicated(['user_id', 'episode', 'treatment']).any():
        raise ValueError('duplicate user/episode/treatment')
    test_users = set(labels.loc[labels.split == 'test', 'user_id'])
    purge = labels.split.ne('test') & labels.user_id.isin(test_users)
    retained, excluded = labels.loc[~purge].copy(), labels.loc[purge].copy()
    user_sets = [set(retained.loc[retained.split == s, 'user_id']) for s in ['train', 'validation', 'test']]
    if any(user_sets[i] & user_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('user overlap between splits')
    return retained, excluded


def select_history(values: np.ndarray, offsets: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError('invalid history values')
    if not np.array_equal(offsets, np.arange(len(offsets)) * 168) or offsets[-1] != len(values):
        raise ValueError('invalid 168-hour episode offsets')
    if len(np.unique(indices)) != len(indices) or (indices < 0).any() or (indices >= len(offsets)-1).any():
        raise ValueError('invalid episode index')
    return values.reshape(-1, 168, 5)[indices].astype(np.float32)


def history_summaries(history: np.ndarray) -> np.ndarray:
    if history.ndim != 3 or history.shape[1:] != (168, 5) or not np.isfinite(history).all():
        raise ValueError('expected finite [episode, 168, 5] histories')
    counts = np.expm1(history.astype(np.float64))
    if (counts < -1e-7).any():
        raise ValueError('history log-counts must be nonnegative')
    return np.concatenate([np.log1p(counts[:, :144].sum(axis=1)),
                           np.log1p(counts[:, 144:].sum(axis=1))], axis=1)
