"""Train dataset-specific LightGCN on time-split binary user/item interactions."""
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from models.lightgcn import LightGCN, normalized_graph, sample_negatives
from utils.paths import PROJECT_ROOT, CAPSTONE_ROOT

ROOT = PROJECT_ROOT
RAW = CAPSTONE_ROOT / 'data/raw/kaggle'
OUT = ROOT / 'artifacts/lightgcn-embeddings'
SEED = 2026


def load(source):
    paths = [RAW / 'electronics/events.csv'] if source == 'electronics' else sorted((RAW / 'cosmetics').glob('20??-*.csv'))
    parts = []
    counts = {'raw': 0, 'sampled': 0, 'removed_action': 0, 'invalid': 0}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=['event_time', 'event_type', 'user_id', 'product_id'],
                                 dtype={'user_id': 'Int64', 'product_id': 'Int64'}, chunksize=400_000):
            counts['raw'] += len(chunk)
            if source == 'cosmetics':
                chunk = chunk[chunk['user_id'].notna() & chunk['user_id'].mod(10).eq(0)].copy()
            counts['sampled'] += len(chunk)
            counts['removed_action'] += int(chunk['event_type'].eq('remove_from_cart').sum())
            chunk = chunk[chunk['event_type'].isin(['view', 'cart', 'purchase'])].copy()
            chunk['time'] = pd.to_datetime(chunk['event_time'], utc=True, errors='coerce')
            valid = chunk[['user_id', 'product_id', 'time']].notna().all(axis=1)
            counts['invalid'] += int((~valid).sum())
            parts.append(chunk.loc[valid, ['user_id', 'product_id', 'time']])
    if not parts:
        raise ValueError('no events')
    events = pd.concat(parts, ignore_index=True)
    events['user_id'] = events['user_id'].astype('int64')
    events['product_id'] = events['product_id'].astype('int64')
    return events, counts


def encode_heldout(frame, users, items, train_keys):
    uid = pd.Index(users).get_indexer(frame['user_id'])
    iid = pd.Index(items).get_indexer(frame['product_id'])
    known = (uid >= 0) & (iid >= 0)
    u, i = uid[known], iid[known]
    allkeys = u.astype(np.int64) * len(items) + i
    novel = ~np.isin(allkeys, train_keys, assume_unique=False)
    eligible = pd.DataFrame({'u': u[novel], 'i': i[novel], 'time': frame.loc[known, 'time'].to_numpy()[novel]})
    eligible = eligible.sort_values('time', kind='stable').drop_duplicates('u')
    return eligible, np.unique(allkeys[novel]), {'events': len(frame), 'known_user_item_events': int(known.sum()),
                      'novel_events': int(novel.sum()), 'eligible_users': len(eligible)}


def evaluate(user_vectors, item_vectors, frame, train_keys, eval_keys, n_items, popularity, seed):
    sample = frame.sample(n=min(2000, len(frame)), random_state=seed)
    u, i = sample.u.to_numpy(), sample.i.to_numpy()
    if not len(u):
        return {'users': 0}
    rng = np.random.default_rng(seed)
    # Exclude all observed positives in this evaluation interval as well as training edges.
    forbidden = np.unique(np.r_[train_keys, eval_keys])
    negatives = sample_negatives(np.repeat(u, 100), n_items, forbidden, rng).reshape(-1, 100)
    candidates = np.column_stack((i, negatives))
    scores = (user_vectors[u, None, :] * item_vectors[candidates]).sum(axis=2)
    pop_scores = popularity[candidates]
    # Random candidate ordering handles identical scores without favoring the positive.
    tie = rng.random(scores.shape) * 1e-9
    def metrics(values):
        ranks = 1 + ((values[:, 1:] + tie[:, 1:]) > (values[:, :1] + tie[:, :1])).sum(axis=1)
        return {'hit_at_10': float((ranks <= 10).mean()),
                'ndcg_at_10': float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean())}
    return {'users': len(u), 'negative_candidates_per_user': 100,
            'lightgcn': metrics(scores), 'popularity': metrics(pop_scores)}


def run(source):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    rng = np.random.default_rng(SEED)
    events, counts = load(source)
    first, end = events.time.min(), events.time.max()
    train_cut = first + (end - first) * .7
    val_cut = first + (end - first) * .85
    train = events[events.time < train_cut]
    validation = events[(events.time >= train_cut) & (events.time < val_cut)]
    test = events[events.time >= val_cut]
    users, u = np.unique(train.user_id.to_numpy(), return_inverse=True)
    items, i = np.unique(train.product_id.to_numpy(), return_inverse=True)
    graph, pairs = normalized_graph(u, i, len(users), len(items))
    keys = np.sort(pairs[:, 0].astype(np.int64) * len(items) + pairs[:, 1])
    val, val_keys, val_coverage = encode_heldout(validation, users, items, keys)
    test_frame, test_keys, test_coverage = encode_heldout(test, users, items, keys)
    popularity = np.bincount(pairs[:, 1], minlength=len(items))
    model = LightGCN(len(users), len(items), dimensions=32, layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    history, best, best_weights, best_epoch = [], -1.0, None, 0
    for epoch in range(1, 41):
        ids = rng.choice(len(pairs), min(len(pairs), 150_000), replace=False)
        batch = pairs[ids]
        neg = sample_negatives(batch[:, 0], len(items), keys, rng)
        user_z, item_z = model(graph)
        bu, bi, bn = (torch.as_tensor(v, dtype=torch.long) for v in (batch[:, 0], batch[:, 1], neg))
        difference = (user_z[bu] * (item_z[bi] - item_z[bn])).sum(dim=1)
        base = model.embedding.weight
        reg = (base[bu].square().sum() + base[bi+len(users)].square().sum() + base[bn+len(users)].square().sum()) / len(batch)
        loss = F.softplus(-difference).mean() + 1e-4 * reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            uz, iz = model(graph)
            metrics = evaluate(uz.numpy(), iz.numpy(), val, keys, val_keys, len(items), popularity, SEED)
        if not metrics['users']:
            raise ValueError('no warm, previously unseen validation interactions')
        history.append({'epoch': epoch, 'loss': float(loss.detach()), 'validation': metrics})
        if metrics['lightgcn']['ndcg_at_10'] > best:
            best = metrics['lightgcn']['ndcg_at_10']
            best_weights = model.embedding.weight.detach().clone()
            best_epoch = epoch
        print(source, 'epoch', epoch, 'loss', round(float(loss.detach()), 4), 'validation', metrics, flush=True)
        if epoch - best_epoch >= 6:
            break
    with torch.no_grad():
        model.embedding.weight.copy_(best_weights)
        uz, iz = model(graph)
        uz, iz = uz.numpy(), iz.numpy()
    test_metrics = evaluate(uz, iz, test_frame, keys, test_keys, len(items), popularity, SEED+1)
    out = OUT / source
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'embeddings.npz', user_ids=users, item_ids=items,
                        user_embeddings=uz.astype('float32'), item_embeddings=iz.astype('float32'))
    result = {'source': source, 'sample': 'all users' if source == 'electronics' else 'user_id modulo 10 == 0',
              'counts': counts, 'first': first.isoformat(), 'train_cut': train_cut.isoformat(),
              'validation_cut': val_cut.isoformat(), 'last': end.isoformat(),
              'train_users': len(users), 'train_items': len(items), 'train_distinct_binary_edges': len(pairs),
              'validation_coverage': val_coverage, 'test_coverage': test_coverage,
              'model': {'dimensions': 32, 'layers': 2, 'epochs_ran': len(history), 'batch_edges_max': 150000,
                        'learning_rate': .01, 'bpr_regularization': 1e-4,
                        'selected_epoch': int(np.argmax([h['validation']['lightgcn']['ndcg_at_10'] for h in history])+1)},
              'history': history, 'test': test_metrics,
              'caveats': ['Binary undirected graph merges view/cart/purchase and excludes removal; not an action-aware GNN.',
                          'Samples unobserved products as negatives; these are not observed dislikes.',
                          'Warm, novel-item, sampled-candidate ranking is not full-catalog ranking; train-unseen users/items are excluded.',
                          'Temporal split; cutoff excludes later interactions from training graph; not yet an evaluation of persona quality.',
                          'No user-specific GRU combination, clusters, item features, causal inference or uplift.',
                          'Saved arrays include pseudonymous raw user and product identifiers; keep local and review permissions before sharing.']}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, 'test', test_metrics, flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'cosmetics'):
            raise ValueError(source)
        run(source)
