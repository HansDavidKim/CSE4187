"""Reproduce the existing masked-GRU pilot and profile held-out input prefixes.

Run with the capstone project root on PYTHONPATH; its data stays in place.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ecommerce_sim.paths import CAPSTONE_ROOT



CAPSTONE = CAPSTONE_ROOT
SOURCE = CAPSTONE / 'data/raw/kaggle/electronics/events.csv'
OUT = Path('artifacts/electronics-personas')


def main():
    from src.irl.fit import load_events

    from src.irl.masked_sequence_personas import fit

    from src.irl.sequence_personas import END, sequence_rows

    OUT.mkdir(parents=True, exist_ok=True)
    report = fit('electronics', SOURCE, OUT / 'electronics_personas.json')
    events, _ = load_events('electronics', SOURCE)
    rows = sequence_rows(events, max_users=12000, max_tokens=64, seed=42)
    assignments = pd.read_parquet(OUT / 'electronics_personas.parquet').set_index('user_id')
    if not assignments.index.is_unique or set(assignments.index) != {row[0] for row in rows}:
        raise ValueError('Saved assignments do not match the sampled users')
    _, test = train_test_split(np.arange(len(rows)), test_size=0.2, random_state=42)
    profiles = []
    for i in test:
        user, tokens, _, sessions, _, _ = rows[i]
        if sessions < 2:
            continue
        profiles.append({'cluster': int(assignments.loc[user, 'cluster']),
                         'views': tokens.count(0), 'carts': tokens.count(1),
                         'purchases': tokens.count(2), 'session_ends': tokens.count(END),
                         'tokens': len(tokens)})
    frame = pd.DataFrame(profiles)
    result = []
    for cluster, group in frame.groupby('cluster'):
        action_count = group[['views', 'carts', 'purchases']].sum()
        result.append({'cluster': int(cluster), 'heldout_users': len(group),
                       'event_share': (action_count / action_count.sum()).to_dict(),
                       'buyer_share': float(group.purchases.gt(0).mean()),
                       'cart_user_share': float(group.carts.gt(0).mean()),
                       'median_tokens': float(group.tokens.median()),
                       'prefix_at_limit_share': float(group.tokens.eq(64).mean())})
    report['heldout_mature_prefix_profiles'] = result
    report['profile_note'] = ('Held-out users with at least two sessions over the full record; '
                              'event counts use the first 64 tokens only. Session eligibility '
                              'uses full history, and reaching 64 tokens is only a truncation proxy.')
    (OUT / 'profiles.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
