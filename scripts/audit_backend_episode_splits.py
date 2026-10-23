#!/usr/bin/env python3
"""Audit user boundaries and cross-episode label/history overlap in Backend-TS."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(labels: pd.DataFrame, histories: dict[str, np.ndarray]) -> dict:
    required = {'split', 'sequence_index', 'user_id', 'episode', 'treatment', 'login_within_3h'}
    if not required.issubset(labels) or labels[list(required)].isna().any().any():
        raise ValueError('missing required labels')
    if labels.duplicated(['episode', 'user_id', 'treatment']).any():
        raise ValueError('duplicate user/episode/treatment')
    if not labels.treatment.isin([0, 1]).all() or not labels.login_within_3h.isin([0, 1]).all():
        raise ValueError('non-binary labels')
    for split, group in labels.groupby('split'):
        values = histories[split]
        if values.ndim != 3 or values.shape[1:] != (168, 5):
            raise ValueError('history must be [episode, 168 hours, 5 methods]')
        indices = group.sequence_index.to_numpy()
        if len(np.unique(indices)) != len(indices) or not np.array_equal(np.sort(indices), np.arange(len(values))):
            raise ValueError('invalid episode indices')
    users = {split: set(group.user_id) for split, group in labels.groupby('split')}
    overlaps = {f'{a}__{b}': len(users[a] & users[b]) for a, b in
                [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    control = labels[labels.treatment == 0]
    treated = labels[labels.treatment == 1]
    paired = control.merge(treated, on=['episode', 'user_id', 'split'], suffixes=('_control', '_treated'), validate='one_to_one')
    matches = sum(bool(histories[row.split][row.sequence_index_treated, :3, 0].sum() > 0)
                  == bool(row.login_within_3h_control) for row in paired.itertuples())
    return {'episodes': len(labels), 'unique_users': labels.user_id.nunique(),
            'user_overlap': overlaps, 'paired_control_and_treated_episodes': len(paired),
            'control_outcome_matches_first_3h_treated_history': int(matches),
            'interpretation': 'Shared information across paired episodes requires user-grouped splits; matching is not proof of counterfactual identity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('artifacts/backend-ts-features'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-split-audit.json'))
    args = parser.parse_args()
    labels = pd.read_csv(args.features / 'episode_labels.csv')
    histories = {}
    for split in ['train', 'validation', 'test']:
        with np.load(args.features / f'{split}_history.npz', allow_pickle=False) as data:
            if data['features'].tolist() != ['LOGIN', 'GET', 'POST', 'PUT', 'DELETE']:
                raise ValueError('unexpected feature order')
            offsets = data['offsets']
            if not np.array_equal(offsets, np.arange(len(offsets)) * 168) or offsets[-1] != len(data['values']):
                raise ValueError('unexpected episode offsets')
            histories[split] = data['values'].reshape(-1, 168, 5)
    result = audit(labels, histories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
