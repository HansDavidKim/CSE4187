"""Daily action states under the explicit assumption of complete platform logs."""
import numpy as np
import pandas as pd


def calendar_days(observed_days):
    """Fill only between each user's first and last observed dates, inclusive.

    Unsupported-action days retain their exclusion flag. This is an assumed
    observed window, not evidence that the user was observable throughout it.
    """
    if observed_days.empty:
        return observed_days.copy()
    days = observed_days.sort_values(['user', 'day']).reset_index(drop=True)
    if days.duplicated(['user', 'day']).any():
        raise ValueError('user/day must be unique')
    spans = days.groupby('user', sort=False)['day'].agg(['min', 'max'])
    lengths = (spans['max'] - spans['min']).dt.days.to_numpy() + 1
    users = np.repeat(spans.index.to_numpy(), lengths)
    starts = np.repeat(spans['min'].dt.tz_localize(None).to_numpy(), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    grid = pd.DataFrame({'user': users, 'day': pd.to_datetime(starts + offsets * np.timedelta64(1, 'D'), utc=True)})
    panel = grid.merge(days, on=['user', 'day'], how='left', validate='one_to_one', indicator=True)
    panel['assumed_inactive'] = panel['_merge'].eq('left_only')
    panel = panel.drop(columns='_merge')
    for column in ('view', 'cart', 'purchase', 'remove', 'action_code'):
        panel[column] = panel[column].fillna(0).astype('int8')
    panel['excluded_remove_day'] = panel['excluded_remove_day'].eq(True)
    panel['eligible'] = ~panel['excluded_remove_day']
    return panel
