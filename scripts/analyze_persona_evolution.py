"""Compare early and later session windows with saved masked-GRU checkpoints."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ecommerce_sim.persona_evolution import session_tokens, summarize_tokens
from src.irl.fit import load_events
from src.irl.masked_sequence_personas import MaskedGRUEncoder, embeddings
from src.irl.sequence_personas import padded, sequence_rows


CAPSTONE = Path('/Users/kimdaewon/Desktop/캡스톤디자인')
OUT = Path('artifacts/persona-evolution')


def classify(model, checkpoint, sequences):
    tokens, lengths = padded(sequences, max_tokens=64)
    z = embeddings(model, tokens, lengths)
    z = (z - checkpoint['scaler_mean'].numpy()) / checkpoint['scaler_scale'].numpy()
    distances = ((z[:, None, :] - checkpoint['centers'].numpy()[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def profile(frame, label):
    count = len(frame)
    if not count:
        return {'users': 0}
    return {'users': count,
            'early_buyer_share': float(frame.early_purchases.gt(0).mean()),
            'later_buyer_share': float(frame.later_purchases.gt(0).mean()),
            'early_cart_share': float(frame.early_carts.gt(0).mean()),
            'later_cart_share': float(frame.later_carts.gt(0).mean()),
            'early_view_event_share': float(frame.early_views.sum() / frame.early_events.sum()),
            'later_view_event_share': float(frame.later_views.sum() / frame.later_events.sum()),
            'early_median_events': float(frame.early_events.median()),
            'later_median_events': float(frame.later_events.median()),
            'early_at_limit_share': float(frame.early_tokens.eq(64).mean()),
            'later_at_limit_share': float(frame.later_tokens.eq(64).mean()),
            'label': label}


def run(source):
    torch.set_num_threads(1)
    checkpoint = torch.load(CAPSTONE / f'data/models/{source}_masked_sequence_personas.pt',
                            map_location='cpu', weights_only=True)
    model = MaskedGRUEncoder()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    events, quality = load_events(source, CAPSTONE / f'data/raw/kaggle/{source}/events.csv')
    rows = sequence_rows(events, 12000, 64, 42)
    _, test = train_test_split(np.arange(len(rows)), test_size=.2, random_state=42)
    saved = pd.read_parquet(CAPSTONE / f'data/models/{source}_masked_sequence_personas.parquet').set_index('user_id')
    if not saved.index.is_unique or set(saved.index) != {r[0] for r in rows}:
        raise ValueError('Reconstructed cohort differs from checkpoint cohort')
    full_labels = classify(model, checkpoint, [r[1] for r in rows])
    if not np.array_equal(full_labels, saved.loc[[r[0] for r in rows], 'cluster'].to_numpy()):
        raise ValueError('Checkpoint does not reproduce saved full-prefix labels')
    test_users = {rows[i][0] for i in test if rows[i][3] >= 4}
    selected = events[events.user.isin(test_users)]
    early, later, cumulative, metadata = [], [], [], []
    for user, group in selected.groupby('user', sort=False):
        sessions = [g.action.tolist() for _, g in group.groupby('session', sort=False)]
        if len(sessions) < 4:
            raise ValueError('repeat-session count mismatch')
        a, b = session_tokens(sessions[:2]), session_tokens(sessions[2:4])
        both = session_tokens(sessions[:4])
        early.append(a)
        later.append(b)
        cumulative.append(both)
        x, y = summarize_tokens(a), summarize_tokens(b)
        metadata.append({**{'early_' + k: v for k, v in x.items()},
                         **{'later_' + k: v for k, v in y.items()},
                         'cumulative_contains_later': len(a) < 64})
    if len(metadata) != len(test_users):
        raise ValueError('Missing selected held-out users')
    frame = pd.DataFrame(metadata)
    frame['early_cluster'] = classify(model, checkpoint, early)
    frame['later_cluster'] = classify(model, checkpoint, later)
    frame['cumulative_cluster'] = classify(model, checkpoint, cumulative)
    k = int(checkpoint['cluster_count'])
    matrix = pd.crosstab(frame.early_cluster, frame.later_cluster).reindex(
        index=range(k), columns=range(k), fill_value=0)
    cum_matrix = pd.crosstab(frame.early_cluster, frame.cumulative_cluster).reindex(
        index=range(k), columns=range(k), fill_value=0)
    changed = frame.early_cluster.ne(frame.later_cluster)
    paths = []
    for (a, b), subset in frame.groupby(['early_cluster', 'later_cluster']):
        paths.append({'early_cluster': int(a), 'later_cluster': int(b),
                      **profile(subset, f'{a}->{b}')})
    result = {'source': source, 'source_quality': quality, 'checkpoint': str(
        CAPSTONE / f'data/models/{source}_masked_sequence_personas.pt'),
        'users': len(frame), 'heldout_user_disjoint_from_encoder_train': True,
        'early_sessions': [1, 2], 'later_sessions': [3, 4],
        'sample_restriction': 'held-out sampled users with >=4 sessions in complete observed record',
        'cluster_count': k, 'early_to_later_counts': matrix.to_numpy().tolist(),
        'early_to_cumulative_counts': cum_matrix.to_numpy().tolist(),
        'changed_early_to_later': int(changed.sum()),
        'changed_early_to_later_share': float(changed.mean()),
        'changed_early_to_cumulative_share': float(frame.early_cluster.ne(frame.cumulative_cluster).mean()),
        'cumulative_contains_later_share': float(frame.cumulative_contains_later.mean()),
        'changed_profile': profile(frame[changed], 'changed'),
        'unchanged_profile': profile(frame[~changed], 'unchanged'),
        'paths': paths,
        'limitations': ['Conditional on 4 observed sessions, not the whole population; selection uses future.',
                        'Separate windows can receive different labels solely because of finite length, shifting observations or model uncertainty.',
                        'Cluster centroids and encoder trained on first-64 full prefixes, not separately on two-session windows.',
                        'Cumulative input is first 64 tokens of first 4 sessions and may not include later sessions.',
                        'Clusters learned separately by dataset; cluster IDs are not equivalent across datasets.',
                        'Session end is inferred and date intervals between windows are unequal.',
                        'Same windows determine clusters and are summarized; no prospective forecast or causal preference change established.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'source': source, 'users': len(frame), 'matrix': result['early_to_later_counts'],
                      'changed': result['changed_early_to_later_share'],
                      'early_to_cumulative_changed': result['changed_early_to_cumulative_share']}, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in {'electronics', 'retailrocket'}:
        raise SystemExit('Specify electronics or retailrocket')
    run(sys.argv[1])
