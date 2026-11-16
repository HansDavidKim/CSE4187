import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location('run_ts_boss', Path(__file__).parents[1] / 'scripts/run_ts_boss.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mpdag_keeps_undirected_and_lagged_edges():
    graph = np.full((2, 2, 2), '', dtype='<U3')
    graph[0, 1, 0] = graph[1, 0, 0] = 'o-o'
    graph[1, 0, 1] = '-->'
    assert module.graph_edges(graph, ['a', 'b']) == [
        {'source': 'a', 'target': 'b', 'lag': 0, 'orientation': 'undirected'},
        {'source': 'b', 'target': 'a', 'lag': 1, 'orientation': 'directed'},
    ]
