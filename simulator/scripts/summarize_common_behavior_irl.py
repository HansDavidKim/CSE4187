"""Create one coefficient observation per fitted dataset/cluster, not per user."""
import json
from pathlib import Path

import pandas as pd

from models.common_behavior_irl import COEFFICIENT_NAMES


def main():
    root = Path('artifacts/common-behavior-irl')
    rows = []
    for source in ('electronics', 'retailrocket', 'cosmetics'):
        result = json.loads((root / f'{source}.json').read_text())
        if tuple(result['feature_names']) != COEFFICIENT_NAMES:
            raise ValueError('Inconsistent feature ordering')
        for group in result['groups']:
            if group['status'] != 'fit':
                continue
            row = {'source': source, 'k': result['k'], 'cluster': group['cluster'],
                   'train_users': group['train_users'],
                   'train_users_with_windows': group['train_users_with_windows'],
                   'train_windows': group['train_windows']}
            for name, coefficient, std in zip(COEFFICIENT_NAMES, group['coefficients'], group['bootstrap_std'], strict=True):
                row['reward_' + name] = coefficient
                row['bootstrap_se_' + name] = std
            rows.append(row)
    table = pd.DataFrame(rows)
    if table.duplicated(['source', 'k', 'cluster']).any() or table.isna().any().any():
        raise ValueError('Invalid group coefficient table')
    table.to_csv(root / 'cluster_coefficients.csv', index=False)
    print(f'{len(table)} cluster coefficient vectors saved')


if __name__ == '__main__':
    main()
