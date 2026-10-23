#!/usr/bin/env python3
"""Create Backend-TS hourly history sequences and separately timed episode labels."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from ecommerce_sim.preprocessing.backend_ts import METHODS, hourly_history

PUSH = {
    'GAME_A_APR': '2022-04-16 12:00:00', 'GAME_A_MAY': '2022-05-14 12:00:00',
    'GAME_A_JUN': '2022-06-18 12:00:00', 'GAME_B_JUN': '2022-06-21 12:30:00',
    'GAME_B_JUL': '2022-07-12 12:30:00', 'GAME_C_JUL': '2022-07-02 12:30:00',
}
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/backend-ts-features'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sequences: dict[str, list[np.ndarray]] = {'train': [], 'validation': [], 'test': []}
    labels = []
    with zipfile.ZipFile(args.archive) as archive:
        for folder in ('GAME_A_APR', 'GAME_A_MAY', 'GAME_A_JUN'):
            info = json.loads(archive.read(f'{folder}/info.json'))
            for record in info:
                user = record['X'].split('_')[0]
                treatment = int(record['T'])
                if treatment not in (0, 1):
                    raise ValueError('invalid treatment label')
                outcome = int(record['Y'][0])
                if outcome not in (0, 1):
                    raise ValueError('invalid outcome label')
                index = pd.Timestamp(PUSH[folder]) - pd.Timedelta(days=7 * (1 - treatment))
                events = pd.read_parquet(io.BytesIO(archive.read(f"{folder}/{record['X']}")))
                history = hourly_history(events, index)
                fraction = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16) / 2**32
                split = 'test' if folder == 'GAME_A_JUN' else ('validation' if fraction < 0.2 else 'train')
                sequences[split].append(history)
                labels.append({'split': split, 'sequence_index': len(sequences[split]) - 1,
                    'user_id': user, 'episode': folder, 'index_time': index.isoformat(),
                    'treatment': treatment, 'login_within_3h': outcome})
    for split, values in sequences.items():
        if not values:
            raise ValueError(f'empty split {split}')
        np.savez_compressed(args.output / f'{split}_history.npz', values=np.concatenate(values),
            offsets=np.arange(len(values) + 1) * 168, features=np.asarray(METHODS))
    pd.DataFrame(labels).to_csv(args.output / 'episode_labels.csv', index=False)
    summary = {'episodes_by_split': {name: len(values) for name, values in sequences.items()},
        'history_features': list(METHODS), 'history_hours': 168,
        'treatment': 'push label; 0 is one-week-earlier pseudo-control, not randomized',
        'outcome': 'login during 3h following index',
        'warning': 'History-only sequences do not contain treatment or outcome; do not claim an intervention DAG.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
