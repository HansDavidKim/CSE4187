#!/usr/bin/env python3
"""Prepare user-session exposure sequences; Pure is diagnostic only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ['is_rand', 'is_click', 'is_like', 'long_view']


def build_sessions(frame: pd.DataFrame) -> list[np.ndarray]:
    needed = ['user_id', 'time_ms', *FEATURES]
    if frame[needed].isna().any().any() or not frame[FEATURES].isin([0, 1]).all().all():
        raise ValueError('missing or nonbinary exposure/feedback')
    frame = frame.sort_values(['user_id', 'time_ms']).copy()
    # Tied timestamps have no reliable order: exclude them rather than inventing one.
    frame = frame.loc[~frame.duplicated(['user_id', 'time_ms'], keep=False)].copy()
    gap = frame.groupby('user_id').time_ms.diff()
    frame['session'] = (gap.isna() | (gap > 30 * 60 * 1000)).cumsum()
    return [(group.user_id.iloc[0], group[FEATURES].to_numpy(dtype=np.float32))
            for _, group in frame.groupby('session', sort=False) if len(group) >= 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--variant', choices=['1k', 'pure'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = [f'log_random_4_22_to_5_08_{args.variant}.csv', f'log_standard_4_22_to_5_08_{args.variant}.csv']
    columns = ['user_id', 'time_ms', 'date', *FEATURES]
    data = pd.concat([pd.read_csv(args.data / name, usecols=columns) for name in names], ignore_index=True)
    # Fixed disjoint user groups plus forward calendar periods; no per-exposure random split.
    group = pd.util.hash_pandas_object(data.user_id, index=False).to_numpy(dtype=np.uint64) % 10
    periods = {'train': (group < 7) & (data.date <= 20220501),
               'validation': (group == 7) & data.date.between(20220502, 20220505),
               'test': (group >= 8) & (data.date >= 20220506)}
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, selected in periods.items():
        frame = data.loc[selected]
        grouped = build_sessions(frame)
        if not grouped:
            raise ValueError(f'no usable sessions in {split}')
        users, sessions = zip(*grouped)
        offsets = np.r_[0, np.cumsum([len(s) for s in sessions])]
        np.savez_compressed(args.output / f'{split}_sessions.npz', values=np.concatenate(sessions),
                            offsets=offsets, features=np.asarray(FEATURES), user_ids=np.asarray(users))
        summary[split] = {'sessions': len(sessions), 'events': int(offsets[-1]),
                          'users_before_session_filter': int(frame.user_id.nunique()),
                          'random_events_before_session_filter': int(frame.is_rand.sum())}
    summary['warning'] = ('Pure has incomplete chronological logs and is diagnostic only' if args.variant == 'pure'
                          else 'is_rand compares recommendation policies, not notification versus no notification')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
