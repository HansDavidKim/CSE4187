"""Binary bipartite LightGCN with BPR and unobserved-item negatives."""
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def normalized_graph(users, items, n_users, n_items):
    pairs = np.unique(np.column_stack((users, items)), axis=0)
    u, v = pairs[:, 0], pairs[:, 1] + n_users
    row, col = np.r_[u, v], np.r_[v, u]
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n_users+n_items, n_users+n_items)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree, dtype=float)
    inverse[degree > 0] = degree[degree > 0] ** -.5
    norm = sp.diags(inverse) @ adj @ sp.diags(inverse)
    coo = norm.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long),
                                   torch.tensor(coo.data, dtype=torch.float32), norm.shape).coalesce(), pairs


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dimensions=32, layers=2):
        super().__init__()
        self.n_users, self.layers = n_users, layers
        self.embedding = nn.Embedding(n_users + n_items, dimensions)
        nn.init.normal_(self.embedding.weight, std=.1)

    def forward(self, graph):
        current = self.embedding.weight
        representations = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(graph, current)
            representations.append(current)
        combined = torch.stack(representations).mean(0)
        return combined[:self.n_users], combined[self.n_users:]


def sample_negatives(users, n_items, train_keys, rng):
    negatives = rng.integers(n_items, size=len(users))
    for _ in range(100):
        keys = users.astype(np.int64) * n_items + negatives
        pos = np.searchsorted(train_keys, keys)
        seen = (pos < len(train_keys)) & (train_keys[np.minimum(pos, len(train_keys)-1)] == keys)
        if not seen.any():
            return negatives
        negatives[seen] = rng.integers(n_items, size=seen.sum())
    raise ValueError('Could not sample an unobserved item; graph is too dense')
