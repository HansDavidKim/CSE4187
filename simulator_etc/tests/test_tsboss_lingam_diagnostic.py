import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('diagnostic', Path(__file__).parents[1] / 'scripts/tsboss_lingam_diagnostic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_order_orientation_preserves_lag_and_skeleton():
    edges = [
        {'source': 'a', 'target': 'b', 'lag': 0, 'orientation': 'directed'},
        {'source': 'a', 'target': 'b', 'lag': 1, 'orientation': 'directed'},
    ]
    result = module.orient_skeleton(edges, ['b', 'a'])
    assert (result[0]['source'], result[0]['target']) == ('b', 'a')
    assert result[1] == edges[1]
    assert edges[0]['source'] == 'a'


def test_hsic_constant_is_undefined():
    assert module.hsic_permutation(np.ones(20), np.arange(20), np.random.default_rng(0)) is None


def test_lingam_order_on_simple_known_non_gaussian_sem():
    rng = np.random.default_rng(42)
    x = rng.laplace(size=3000)
    y = 1.5 * x + rng.laplace(size=3000)
    model = module.DirectLiNGAM(measure='pwling')
    model.fit(np.column_stack([x, y]))
    assert list(model._causal_order) == [0, 1]
