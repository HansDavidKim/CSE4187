"""Load observed events and retain data-quality audit counts."""
import pandas as pd

from utils.paths import CAPSTONE_ROOT

CAPSTONE = CAPSTONE_ROOT


def load_events(source):
    root = CAPSTONE / 'data/raw/kaggle'
    paths = sorted((root / 'cosmetics').glob('20??-*.csv')) if source == 'cosmetics' else [root / source / 'events.csv']
    if not paths:
        raise ValueError('no input files')
    retail = source == 'retailrocket'
    columns = ['visitorid', 'timestamp', 'event'] if retail else ['user_id', 'event_time', 'event_type', 'user_session']
    user_col, time_col, action_col = columns[:3]
    mapping = {'view': 'view', 'addtocart': 'cart', 'transaction': 'purchase'} if retail else {
        'view': 'view', 'cart': 'cart', 'purchase': 'purchase', 'remove_from_cart': 'remove'}
    parts = []
    audit = {'raw_rows': 0, 'sample_rows': 0, 'missing_session_rows': 0,
             'valid_rows_missing_session': 0, 'invalid_user_time_action_rows': 0}
    for path in paths:
        for chunk in pd.read_csv(path, usecols=columns, dtype={user_col: 'string'}, chunksize=500_000):
            audit['raw_rows'] += len(chunk)
            if source == 'cosmetics':
                uid = pd.to_numeric(chunk[user_col], errors='coerce').astype('Int64')
                chunk = chunk.loc[uid.mod(100).eq(0).fillna(False)].copy()
            audit['sample_rows'] += len(chunk)
            at = pd.to_datetime(chunk[time_col], unit='ms' if retail else None, utc=True, errors='coerce')
            action = chunk[action_col].map(mapping)
            valid = chunk[user_col].notna() & at.notna() & action.notna()
            audit['invalid_user_time_action_rows'] += int((~valid).sum())
            if not retail:
                missing = chunk['user_session'].isna()
                audit['missing_session_rows'] += int(missing.sum())
                audit['valid_rows_missing_session'] += int((missing & valid).sum())
            parts.append(pd.DataFrame({'user': chunk.loc[valid, user_col], 'at': at[valid],
                                       'action': action[valid]}))
    if not parts:
        raise ValueError('no events')
    return pd.concat(parts, ignore_index=True), audit
