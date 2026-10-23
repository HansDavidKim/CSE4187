"""Compare sessions 1-2 with 3-4 for held-out users, within each user."""
import json
import sys

import numpy as np
import pandas as pd
from ecommerce_sim.paths import PROJECT_ROOT, CAPSTONE_ROOT


CAPSTONE = CAPSTONE_ROOT
OUT = PROJECT_ROOT / 'artifacts/early-later-two-datasets'


def summarize(events, actions):
    frame = events.copy()
    frame['rank'] = frame.groupby('user', sort=False)['session'].transform(lambda s: pd.factorize(s, sort=False)[0])
    frame = frame[frame['rank'] < 4].copy()
    frame['period'] = np.where(frame['rank'] < 2, 'early', 'later')
    flags = pd.crosstab([frame['user'], frame['period']], frame['action']).reindex(columns=range(len(actions)), fill_value=0)
    flags.columns = actions
    early = flags.xs('early', level='period').sort_index()
    later = flags.xs('later', level='period').reindex(early.index)
    if later.isna().any().any():
        raise ValueError('each user must have four sessions')
    early_view_only = early['view'].gt(0) & early.drop(columns='view').eq(0).all(axis=1)
    later_view_only = later['view'].gt(0) & later.drop(columns='view').eq(0).all(axis=1)
    early_cartbuy = early['cart'].gt(0) | early['purchase'].gt(0)
    later_cartbuy = later['cart'].gt(0) | later['purchase'].gt(0)
    categories = {'early_view_only': early_view_only, 'later_view_only': later_view_only,
                  'early_view_only_later_cart_or_purchase': early_view_only & later_cartbuy,
                  'early_cart_or_purchase_later_view_only': early_cartbuy & later_view_only,
                  'view_only_both_windows': early_view_only & later_view_only,
                  'cart_or_purchase_both_windows': early_cartbuy & later_cartbuy}
    n = len(early)
    outcomes = {}
    for action in actions:
        a, b = early[action].gt(0), later[action].gt(0)
        outcomes[action] = {'early_users': int(a.sum()), 'later_users': int(b.sum()),
                            'neither_then_yes': int((~a & b).sum()),
                            'yes_then_neither': int((a & ~b).sum())}
    return {'users': n, 'counts': {k: int(v.sum()) for k, v in categories.items()},
            'early_view_only_to_later_cart_or_purchase_fraction':
                float((early_view_only & later_cartbuy).sum() / early_view_only.sum()) if early_view_only.sum() else None,
            'actions': outcomes,
            'early_median_events': float(early.sum(axis=1).median()),
            'later_median_events': float(later.sum(axis=1).median())}


def run(source):
    from src.analysis.masked_embedding_clusters import rows_from_events, source_data

    path = CAPSTONE / ('data/raw/kaggle/cosmetics' if source == 'cosmetics' else 'data/raw/kaggle/electronics/events.csv')
    events, quality, actions = source_data(source, path, sample_modulus=100)
    rows = rows_from_events(events, 12000, 64, len(actions), 42)
    emb = pd.read_parquet(CAPSTONE / f'data/models/{source}_masked_embedding_embeddings.parquet')
    if emb.user_id.tolist() != [r[0] for r in rows]:
        raise ValueError('saved embedding cohort differs')
    order = np.random.default_rng(42).permutation(len(rows))
    heldout = order[int(.8 * len(rows)):]
    users = {rows[i][0] for i in heldout if rows[i][2] >= 4}
    result = summarize(events[events.user.isin(users)], actions)
    if result['users'] != len(users):
        raise ValueError('four-session user cohort mismatch')
    result.update({'source': source, 'cohort': 'held-out users in existing 12000 stratified masked-GRU cohort with >=4 observed sessions',
                   'actions_mapping': actions, 'source_quality': quality,
                   'limitations': ['Observed first sessions may not be true onboarding.',
                       'Four sessions required and sampled by total sessions: selected retrospectively, not whole population.',
                       'No proof of causal change, stable preference, uplift or return after observation end.',
                       'Two windows use complete event sequences; prior saved embeddings used first 64 tokens.',
                       'Separate window GRU embeddings unavailable for cosmetics checkpoint in existing files.']})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(source, json.dumps({k: result[k] for k in ('users', 'counts', 'actions')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'cosmetics'):
            raise ValueError(source)
        run(source)
