import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np

from ecommerce_sim.nts_training import default_prior_knowledge, normalization_parameters

spec = importlib.util.spec_from_file_location('prepare_kuairand', Path(__file__).parents[1] / 'scripts/prepare_kuairand.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_sessions_do_not_cross_users_or_ambiguous_timestamp():
    rows = [(user, ts, 0, 1, 0, 0) for user, ts in
            [(1, 100), (1, 200), (1, 300), (1, 400), (1, 400), (2, 100), (2, 200), (2, 300)]]
    frame = pd.DataFrame(rows, columns=['user_id', 'time_ms', *module.FEATURES])
    sessions = module.build_sessions(frame)
    assert [user for user, _ in sessions] == [1, 2]
    assert [len(values) for _, values in sessions] == [3, 3]


def test_feedback_cannot_be_contemporaneous_cause_of_exposure():
    prior = default_prior_knowledge(module.FEATURES)
    assert {(p['from_node'], p['to_node']) for p in prior} == {
        ('is_click', 'is_rand'), ('is_like', 'is_rand'), ('long_view', 'is_rand')}


def test_binary_features_are_left_unscaled_when_requested():
    values = np.array([[0., 2.], [1., 4.]])
    mean, scale, _ = normalization_parameters(values, preserve_binary=True)
    np.testing.assert_array_equal(mean, [0., 3.])
    np.testing.assert_array_equal(scale, [1., 1.])
