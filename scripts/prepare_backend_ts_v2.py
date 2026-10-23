#!/usr/bin/env python3
"""Build user-disjoint Backend-TS feature arrays from validated episode histories."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_sim.preprocessing.backend_ts import METHODS
from ecommerce_sim.preprocessing.backend_ts_v2 import disjoint_split, select_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/backend-ts-features'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-features-v2'))
    args = parser.parse_args()
    labels = pd.read_csv(args.input / 'episode_labels.csv')
    retained, excluded = disjoint_split(labels)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'excluded_earlier_episodes_for_user_disjoint_test': len(excluded), 'splits': {}}
    output_labels = []
    for split in ['train', 'validation', 'test']:
        rows = retained.loc[retained.split == split].copy().reset_index(drop=True)
        if rows.empty:
            raise ValueError(f'empty split: {split}')
        with np.load(args.input / f'{split}_history.npz', allow_pickle=False) as data:
            if data['features'].tolist() != list(METHODS):
                raise ValueError('unexpected feature ordering')
            history = select_history(data['values'], data['offsets'], rows.sequence_index.to_numpy())
        rows['original_sequence_index'] = rows.sequence_index
        rows['sequence_index'] = np.arange(len(rows))
        rows['index_time'] = pd.to_datetime(rows.index_time, errors='raise')
        rows['history_start'] = rows.index_time - pd.Timedelta(days=7)
        rows['outcome_end_exclusive'] = rows.index_time + pd.Timedelta(hours=3)
        rows['pseudo_control'] = rows.treatment.eq(0)
        rows['pair_id'] = rows.episode + ':' + rows.user_id.astype(str)
        rows['episode_id'] = rows.pair_id + ':' + rows.treatment.astype(str)
        output_labels.append(rows)
        count = len(rows)
        np.savez_compressed(args.output / f'{split}.npz', history=history,
            history_grid_mask=np.ones(history.shape, dtype=bool),
            treatment=rows.treatment.to_numpy(dtype=np.uint8),
            outcome_y3=rows.login_within_3h.to_numpy(dtype=np.uint8),
            endpoint_mask=np.ones((count, 2), dtype=bool),
            user_ids=rows.user_id.to_numpy(dtype=str), features=np.asarray(METHODS),
            values=history.reshape(-1, 5), offsets=np.arange(count + 1) * 168)
        result['splits'][split] = {'episodes': count, 'users': rows.user_id.nunique(),
            'treated': int(rows.treatment.sum()), 'control': int((1 - rows.treatment).sum()),
            'login_3h': int(rows.login_within_3h.sum())}
    combined = pd.concat(output_labels, ignore_index=True)
    combined.to_csv(args.output / 'episode_labels.csv', index=False)
    result['feature_roles'] = {'history': list(METHODS), 'treatment': 'treatment', 'outcome': 'outcome_y3'}
    result['warning'] = ['T=0 is a pseudo-control, not randomized.',
        'Grid mask is not evidence of source coverage or verified push delivery.',
        'Episodes were selected based on a pre-index login; results are not for all users.',
        'Labels are episode-level; current NTS accepts only history, not endpoint T/Y.',
        'User-disjoint future test changes the target population compared with original benchmark.']
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
