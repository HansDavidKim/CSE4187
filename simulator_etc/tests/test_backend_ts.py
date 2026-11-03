import numpy as np
import pandas as pd
import pytest
import importlib.util
from pathlib import Path

from preprocessing.backend_ts import hourly_history
from preprocessing.backend_ts_v2 import disjoint_split, select_history
from models.nts_training import bootstrap_user_indices


def test_hourly_history_excludes_index_and_preserves_method():
    index = pd.Timestamp('2022-04-16 12:00')
    events = pd.DataFrame({'timestamp': [index - pd.Timedelta(days=7), index - pd.Timedelta(seconds=1)],
                           'method': ['LOGIN', 'GET']})
    values = hourly_history(events, index)
    assert values.shape == (168, 5)
    assert values[0, 0] == pytest.approx(np.log(2))
    assert values[-1, 1] == pytest.approx(np.log(2))
    events.loc[1, 'timestamp'] = index
    with pytest.raises(ValueError, match='outside'):
        hourly_history(events, index)


def test_split_purges_all_train_episodes_of_test_user_and_keeps_pairs_together():
    labels = pd.DataFrame([
        ['a', 'apr', 'train', 0, 0, 1, '2022-04-09'],
        ['a', 'apr', 'train', 1, 1, 0, '2022-04-16'],
        ['a', 'jun', 'test', 0, 1, 1, '2022-06-18'],
        ['b', 'apr', 'train', 2, 0, 0, '2022-04-09'],
        ['b', 'apr', 'train', 3, 1, 1, '2022-04-16'],
        ['c', 'apr', 'validation', 0, 0, 0, '2022-04-09'],
    ], columns=['user_id', 'episode', 'split', 'sequence_index', 'treatment', 'login_within_3h', 'index_time'])
    retained, excluded = disjoint_split(labels)
    assert len(excluded) == 2 and set(excluded.user_id) == {'a'}
    assert retained[retained.user_id == 'b'].treatment.tolist() == [0, 1]
    assert len(set(retained[retained.split == 'train'].user_id) & set(retained[retained.split == 'test'].user_id)) == 0


def test_history_selection_preserves_episodes_and_rejects_bad_offsets():
    values = np.arange(2 * 168 * 5, dtype=float).reshape(-1, 5)
    selected = select_history(values, np.array([0, 168, 336]), np.array([1]))
    np.testing.assert_array_equal(selected[0], values[168:])
    with pytest.raises(ValueError, match='offsets'):
        select_history(values, np.array([0, 167, 336]), np.array([1]))


def test_outcome_baseline_summaries_have_disjoint_windows():
    spec = importlib.util.spec_from_file_location('backend_baseline', Path(__file__).parents[1] / 'scripts/run_backend_outcome_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    history = np.zeros((1, 168, 5))
    history[0, 143, 0] = np.log1p(2)
    history[0, 144, 0] = np.log1p(3)
    result = module.history_summaries(history)
    assert result[0, 0] == pytest.approx(np.log1p(2))
    assert result[0, 5] == pytest.approx(np.log1p(3))


def test_bootstrap_samples_user_episode_groups_together():
    users = np.array(['a', 'a', 'b', 'b', 'b', 'c'])
    indices = bootstrap_user_indices(users, 10, 42)
    multiplicities = np.bincount(indices, minlength=len(users))
    assert multiplicities[0] == multiplicities[1]
    assert multiplicities[2] == multiplicities[3] == multiplicities[4]
    assert len(bootstrap_user_indices(users, 10, 42)) == len(indices)
