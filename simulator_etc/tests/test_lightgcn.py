import numpy as np
import torch
from models.lightgcn import LightGCN, normalized_graph, sample_negatives
from scripts.train_lightgcn_embeddings import encode_heldout
import pandas as pd


def test_graph_normalization_propagation_and_negatives():
    graph, pairs = normalized_graph(np.array([0, 0, 1, 0]), np.array([0, 1, 1, 0]), 2, 3)
    assert len(pairs) == 3
    assert torch.allclose(graph.to_dense(), graph.to_dense().T)
    model = LightGCN(2, 3, dimensions=4, layers=2)
    users, items = model(graph)
    assert users.shape == (2, 4) and items.shape == (3, 4)
    keys = np.sort(pairs[:, 0] * 3 + pairs[:, 1])
    u = np.array([0, 1] * 100)
    neg = sample_negatives(u, 3, keys, np.random.default_rng(1))
    assert not np.isin(u * 3 + neg, keys).any()


def test_heldout_filters_cold_and_seen_edges():
    frame = pd.DataFrame({'user_id': [10, 10, 20, 99], 'product_id': [100, 101, 101, 100],
                          'time': pd.to_datetime(['2020-01-01'] * 4, utc=True)})
    rows, keys, stats = encode_heldout(frame, np.array([10, 20]), np.array([100, 101]), np.array([0, 3]))
    assert stats['known_user_item_events'] == 3
    assert stats['eligible_users'] == 1
    assert rows.u.tolist() == [0] and keys.tolist() == [1]
