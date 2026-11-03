"""Past-only feature selection for next-action prediction and session NTS exploration."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils.paths import CAPSTONE_ROOT, PROJECT_ROOT

ACTIONS = ('view', 'cart', 'purchase', 'remove_from_cart')
SOURCE_COLUMNS = ['user_id', 'user_session', 'event_time', 'event_type', 'product_id', 'category_id', 'brand', 'price']
TIMEOUT = pd.Timedelta(minutes=30)


def order_events(data):
    """Collapse unambiguous same-action batches; cut sequences at mixed ties."""
    data = data.sort_values(['user_id', 'event_time'], kind='stable').copy()
    keys = ['user_id', 'event_time']
    grouped = data.groupby(keys, sort=False)
    tied = data.duplicated(keys, keep=False)
    batch = tied & grouped.event_type.transform('nunique').eq(1) & grouped.user_session.transform('nunique').eq(1)
    data['batch_size'] = grouped.event_type.transform('size')
    data.loc[batch, ['product_id', 'category_id', 'brand', 'price']] = pd.NA
    repeated_batch = batch & data.duplicated(keys)
    batches_removed = int(repeated_batch.sum())
    data = data.loc[~repeated_batch].copy()
    ambiguous = data.duplicated(keys, keep=False)
    ambiguous_counts = data.loc[ambiguous, 'event_type'].value_counts().to_dict()
    data['order_block'] = ambiguous.groupby(data.user_id).cumsum()
    data['safe_start'] = ~(ambiguous.shift(fill_value=False) & data.user_id.eq(data.user_id.shift()))
    data['safe_end'] = ~(ambiguous.shift(-1, fill_value=False) & data.user_id.eq(data.user_id.shift(-1)))
    return data.loc[~ambiguous].copy(), batches_removed, ambiguous_counts


def load_events(source, modulus):
    root = CAPSTONE_ROOT / 'data/raw/kaggle'
    files = sorted((root / 'cosmetics').glob('20??-*.csv')) if source == 'cosmetics' else [root / 'electronics/events.csv']
    if not files or modulus < 1:
        raise ValueError('missing input files or invalid sample modulus')
    pieces = []
    for path in files:
        for chunk in pd.read_csv(path, usecols=SOURCE_COLUMNS, dtype={'user_id': 'string', 'user_session': 'string', 'product_id': 'string', 'category_id': 'string', 'brand': 'string'}, chunksize=500_000):
            hashes = pd.util.hash_pandas_object(chunk.user_id, index=False).to_numpy()
            pieces.append(chunk.loc[hashes % modulus == 0].copy())
    data = pd.concat(pieces, ignore_index=True)
    data['event_time'] = pd.to_datetime(data.event_time, utc=True, errors='coerce')
    data['price'] = pd.to_numeric(data.price, errors='coerce').where(lambda x: x >= 0)
    initial = len(data)
    data = data.dropna(subset=['user_id', 'user_session', 'event_time'])
    data = data[data.event_type.isin(ACTIONS)].copy()
    before_ties = len(data)
    data, batched_rows, ambiguous_counts = order_events(data)
    if data.empty:
        raise ValueError('no valid events after filtering')
    data = data.sort_values(['user_id', 'event_time'], kind='stable').reset_index(drop=True)
    return data, {'sampled_rows': initial, 'invalid_rows': initial - before_ties,
                  'same_action_batch_rows_collapsed': batched_rows,
                  'ambiguous_tied_rows_removed': before_ties - len(data) - batched_rows,
                  'ambiguous_tied_rows_by_action': ambiguous_counts,
                  'purchase_batches': int(data.event_type.eq('purchase').sum()),
                  'purchase_batches_with_multiple_events': int((data.event_type.eq('purchase') & data.batch_size.gt(1)).sum()),
                  'sample_modulus': modulus, 'files': [str(p) for p in files]}


def make_panel(data, observation_end):
    """Construct decisions from strictly earlier events and conservative inferred ENDs."""
    data = data.sort_values(['user_id', 'event_time'], kind='stable').copy()
    if 'batch_size' not in data:
        data['batch_size'] = 1
    by_user = data.groupby('user_id', sort=False)
    if 'order_block' not in data:
        data['order_block'] = 0
        data['safe_end'] = True
        data['safe_start'] = True
    segment_start = (data.user_session.ne(by_user.user_session.shift()) | data.user_id.ne(data.user_id.shift())
                     | data.order_block.ne(by_user.order_block.shift()))
    data['segment'] = segment_start.cumsum()
    # Item histories exclude ambiguous timestamp blocks and never use the next event.
    for key, label in (('product_id', 'item'), ('category_id', 'category')):
        histories = data.groupby(['user_id', 'order_block', key], sort=False, dropna=True)
        data[f'{label}_prior_count'] = histories.cumcount()
        data[f'{label}_recency_minutes'] = (data.event_time - histories.event_time.shift()).dt.total_seconds() / 60
    for action in ('view', 'purchase'):
        indicator = data.event_type.eq(action).astype(int)
        data[f'item_prior_{action}_count'] = indicator.groupby([data.user_id, data.order_block, data.product_id], dropna=True).cumsum() - indicator
    session = data.groupby('segment', sort=False)
    summaries = session.agg(user_id=('user_id', 'first'), order_block=('order_block', 'first'),
                            safe_end=('safe_end', 'last'), safe_start=('safe_start', 'first'), start=('event_time', 'min'),
                            last=('event_time', 'max'), event_count=('event_type', 'size'),
                            unique_products=('product_id', 'nunique'), unique_categories=('category_id', 'nunique'),
                            mean_event_price=('price', 'mean'))
    for action in ACTIONS:
        summaries[action] = data.event_type.eq(action).groupby(data.segment).sum()
    summaries['duration_minutes'] = (summaries['last'] - summaries['start']).dt.total_seconds() / 60
    previous = summaries.groupby(['user_id', 'order_block'], sort=False)
    for col in ('last', 'duration_minutes', 'event_count', *ACTIONS):
        summaries['previous_' + col] = previous[col].shift()
        summaries.loc[~previous.safe_start.shift().eq(True), 'previous_' + col] = np.nan
    summaries['previous_gap_hours'] = (summaries.start - summaries.previous_last).dt.total_seconds() / 3600
    summaries['session_number'] = previous.cumcount()
    end = summaries.copy()
    end['event_time'] = end['last'] + TIMEOUT
    next_start = previous.start.shift(-1)
    # Do not invent END when a later session starts before the timeout.
    end = end[end.safe_end & (end.event_time < observation_end) & (next_start.isna() | (next_start > end.event_time))]
    ends = pd.DataFrame({'user_id': end.user_id, 'segment': end.index, 'event_time': end.event_time,
                         'event_type': 'end'})
    decisions = pd.concat([data, ends], ignore_index=True).sort_values(['user_id', 'event_time'], kind='stable')
    g = decisions.groupby('segment', sort=False)
    panel = pd.DataFrame(index=decisions.index)
    panel['user_id'] = decisions.user_id
    panel['outcome_at'] = decisions.event_time
    panel['at'] = g.event_time.shift().fillna(decisions.event_time)
    panel['target'] = decisions.event_type
    panel['segment'] = decisions.segment
    panel['session_event_index'] = g.cumcount()
    panel['session_elapsed_minutes'] = (panel['at'] - g.event_time.transform('min')).dt.total_seconds() / 60
    for lag in (1, 2, 3):
        previous_action = g.event_type.shift(lag)
        panel[f'has_previous_action_{lag}'] = previous_action.notna().astype(int)
        for action in ACTIONS:
            panel[f'lag{lag}_{action}'] = previous_action.eq(action).astype(int)
    prior1, prior2 = g.event_type.shift(1), g.event_type.shift(2)
    for earlier in ACTIONS:
        for later in ACTIONS:
            panel[f'prior_pair_{earlier}_{later}'] = (prior2.eq(earlier) & prior1.eq(later)).astype(int)
    panel['previous_inter_event_gap_minutes'] = (g.event_time.shift(1) - g.event_time.shift(2)).dt.total_seconds() / 60
    for col in ('item_prior_count', 'item_prior_view_count', 'item_prior_purchase_count', 'item_recency_minutes', 'category_prior_count', 'category_recency_minutes'):
        panel['previous_' + col] = g[col].shift()
    prev_category, prev2_category = g.category_id.shift(1), g.category_id.shift(2)
    panel['previous_category_switch'] = prev_category.astype('string').ne(prev2_category.astype('string')).astype(float)
    for action in ACTIONS:
        indicator = decisions.event_type.eq(action).astype(int)
        panel['session_' + action + '_count'] = indicator.groupby(decisions.segment).cumsum() - indicator
    # Prior observed event diversity, not a reconstruction of the current cart.
    for action in ('view', 'cart'):
        for col in ('product_id', 'category_id'):
            first = (decisions.event_type.eq(action) & decisions[col].notna()
                     & ~decisions.duplicated(['segment', 'event_type', col]))
            indicator = first.astype(int)
            panel[f'session_{action}_unique_{col}'] = indicator.groupby(decisions.segment).cumsum() - indicator
        valid_price = decisions.event_type.eq(action) & decisions.price.notna()
        price = decisions.price.where(valid_price, 0)
        count = valid_price.astype(int)
        prior_sum = price.groupby(decisions.segment).cumsum() - price
        prior_count = count.groupby(decisions.segment).cumsum() - count
        panel[f'session_{action}_event_price_sum'] = prior_sum
        panel[f'session_{action}_event_price_mean'] = prior_sum.div(prior_count.where(prior_count.gt(0)))
    panel['observed_cart_event_balance'] = panel.session_cart_count - panel.session_remove_from_cart_count
    panel['previous_price_vs_session_view_mean'] = g.price.shift().div(panel.session_view_event_price_mean.where(panel.session_view_event_price_mean.gt(0)))
    purchase_price = decisions.price.where(decisions.event_type.eq('purchase') & decisions.price.notna(), 0)
    purchase_price_count = (decisions.event_type.eq('purchase') & decisions.price.notna()).astype(int)
    purchase_sum = purchase_price.groupby(decisions.segment).cumsum() - purchase_price
    purchase_count = purchase_price_count.groupby(decisions.segment).cumsum() - purchase_price_count
    panel['session_past_purchase_minus_view_price'] = purchase_sum.div(purchase_count.where(purchase_count.gt(0))) - panel.session_view_event_price_mean
    # Batch size counts logged rows, not validated units or distinct items.
    purchase_rows = decisions.batch_size.fillna(0) * decisions.event_type.eq('purchase')
    panel['session_purchase_event_count'] = purchase_rows.groupby(decisions.segment).cumsum() - purchase_rows
    prior_batch_size = g.batch_size.shift().fillna(0)
    panel['previous_purchase_batch_size'] = prior_batch_size.where(g.event_type.shift().eq('purchase'), 0)
    for col in ('price', 'brand', 'category_id', 'product_id'):
        prior = g[col].shift()
        panel['previous_' + col + '_missing'] = prior.isna().astype(int)
        if col == 'price':
            panel['previous_price'] = prior
    for col in ('previous_gap_hours', 'previous_duration_minutes', 'previous_event_count', 'session_number',
                'previous_view', 'previous_cart', 'previous_purchase', 'previous_remove_from_cart'):
        panel[col] = decisions.segment.map(summaries[col])
    panel['previous_session_exists'] = panel.previous_event_count.notna().astype(int)
    panel['history_truncated'] = decisions.segment.map(summaries.order_block).gt(0).astype(int)
    panel['session_left_truncated'] = (~decisions.segment.map(summaries.safe_start).astype(bool)).astype(int)
    panel['hour_sin'] = np.sin(2 * np.pi * (panel['at'].dt.hour + panel['at'].dt.minute / 60) / 24)
    panel['hour_cos'] = np.cos(2 * np.pi * (panel['at'].dt.hour + panel['at'].dt.minute / 60) / 24)
    panel['weekday'] = panel['at'].dt.dayofweek
    # The first event establishes session entry; only subsequent choices are modeled.
    return panel.loc[panel.session_event_index > 0].reset_index(drop=True), summaries.reset_index()


def split_panel(panel):
    times = panel['at']
    start, end = times.min(), times.max()
    train_end, valid_end = start + (end - start) * .6, start + (end - start) * .8
    bucket = pd.util.hash_pandas_object('split:' + panel.user_id.astype(str), index=False).to_numpy() % 10
    outcome = panel['outcome_at']
    masks = [(bucket < 6) & (times < train_end) & (outcome < train_end),
             (bucket == 6) & (times >= train_end) & (outcome < valid_end),
             (bucket >= 7) & (times >= valid_end)]
    return masks, {'start': str(start), 'train_end': str(train_end), 'valid_end': str(valid_end), 'end': str(end)}


def purchase_metrics(actual, prediction):
    actual = np.asarray(actual, dtype=int)
    prediction = np.asarray(prediction, dtype=float)
    return {'rows': len(actual), 'purchases': int(actual.sum()),
            'actual_rate': float(actual.mean()), 'predicted_mean': float(prediction.mean()),
            'rmse': float(np.sqrt(np.mean((prediction - actual) ** 2))),
            'log_loss': float(log_loss(actual, prediction, labels=[0, 1])),
            'average_precision': float(average_precision_score(actual, prediction)) if actual.sum() else None,
            'roc_auc': float(roc_auc_score(actual, prediction)) if np.unique(actual).size == 2 else None}


def predict_purchase(panel):
    """Select using validation RMSE; evaluate once on a user- and time-held-out test."""
    masks, bounds = split_panel(panel)
    exclude = ('user_id', 'at', 'outcome_at', 'target', 'segment')
    x = panel.drop(columns=list(exclude)).replace([np.inf, -np.inf], np.nan)
    parts = [panel.loc[mask] for mask in masks]
    y = [part.target.eq('purchase').astype(int).to_numpy() for part in parts]
    if any(len(part) == 0 for part in parts) or len(np.unique(y[0])) < 2:
        raise ValueError('insufficient train/validation/test rows or training purchases')
    medians = x.loc[masks[0]].median().fillna(0)
    matrices = [x.loc[mask].fillna(medians).to_numpy(dtype=float) for mask in masks]
    scaler = StandardScaler().fit(matrices[0])
    scaled = [scaler.transform(matrix) for matrix in matrices]
    candidates = []
    for c in (.01, .1, 1.):
        model = LogisticRegression(penalty='l1', solver='liblinear', C=c, max_iter=2000, random_state=42)
        model.fit(scaled[0], y[0])
        prob = model.predict_proba(scaled[1])[:, 1]
        candidates.append((f'l1_C={c}', model, True, purchase_metrics(y[1], prob)))
    for leaves in (7, 15):
        model = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=leaves,
                                                min_samples_leaf=50, l2_regularization=1., random_state=42)
        model.fit(matrices[0], y[0])
        prob = model.predict_proba(matrices[1])[:, 1]
        candidates.append((f'histgb_leaves={leaves}', model, False, purchase_metrics(y[1], prob)))
    baseline = float(y[0].mean())
    name, model, use_scaled, _ = min(candidates, key=lambda candidate: candidate[3]['rmse'])
    prediction = model.predict_proba((scaled if use_scaled else matrices)[2])[:, 1]
    reliability = []
    bins = pd.qcut(pd.Series(prediction), q=10, duplicates='drop')
    summary = pd.DataFrame({'actual': y[2], 'prediction': prediction, 'bin': bins})
    for _, group in summary.groupby('bin', observed=True):
        reliability.append({'rows': len(group), 'actual_rate': float(group.actual.mean()),
                            'predicted_mean': float(group.prediction.mean())})
    result = {'split': bounds, 'selected_model': name,
              'validation_models': {candidate[0]: candidate[3] for candidate in candidates},
              'validation_baseline': purchase_metrics(y[1], np.full(len(y[1]), baseline)),
              'test_model': purchase_metrics(y[2], prediction),
              'test_baseline': purchase_metrics(y[2], np.full(len(y[2]), baseline)),
              'test_calibration_bins': reliability,
              'users': {name: int(part.user_id.nunique()) for name, part in zip(('train', 'validation', 'test'), parts)},
              'train_purchase_rate': baseline,
              'target': 'next observed action is purchase; session end inferred after timeout',
              'warning': 'Observational next-action prediction, not treatment effects or trajectory simulation.'}
    return result, pd.DataFrame({'actual': y[2], 'prediction': prediction})


def select_features(panel):
    masks, bounds = split_panel(panel)
    feature_names = [c for c in panel if c not in ('user_id', 'at', 'outcome_at', 'target', 'segment')]
    x = panel[feature_names].replace([np.inf, -np.inf], np.nan)
    train, valid, test = [panel.loc[m] for m in masks]
    if min(len(train), len(valid), len(test)) == 0 or train.target.nunique() < 2:
        raise ValueError('insufficient data for user-disjoint temporal evaluation')
    missing = x.loc[masks[0]].isna().mean()
    medians = x.loc[masks[0]].median().fillna(0)
    x = x.fillna(medians)
    transforms = {}
    purchase_train = train.target.eq('purchase').astype(int)
    purchase_valid = valid.target.eq('purchase').astype(int)
    if purchase_train.nunique() == 2:
        for name in x.columns:
            values = x[name]
            if values.loc[masks[0]].nunique() < 10 or values.loc[masks[0]].min() < 0:
                continue
            positive = values.loc[masks[0] & values.gt(0)]
            tau = float(positive.median()) if len(positive) else 1.
            choices = {'linear': values, 'log1p': np.log1p(values),
                       'saturating_exp': -np.expm1(-values.clip(lower=0) / tau)}
            scores = {}
            for kind, candidate in choices.items():
                scaler1 = StandardScaler().fit(candidate.loc[masks[0]].to_numpy().reshape(-1, 1))
                fitted = LogisticRegression(max_iter=1000).fit(scaler1.transform(candidate.loc[masks[0]].to_numpy().reshape(-1, 1)), purchase_train)
                probability = fitted.predict_proba(scaler1.transform(candidate.loc[masks[1]].to_numpy().reshape(-1, 1)))
                scores[kind] = float(log_loss(purchase_valid, probability, labels=[0, 1]))
            chosen = min(scores, key=scores.get)
            x[name] = choices[chosen]
            transforms[name] = {'chosen': chosen, 'tau': tau, 'validation_log_loss': scores}
    keep = x.loc[masks[0]].nunique() > 1
    x = x.loc[:, keep]
    scaler = StandardScaler().fit(x.loc[masks[0]])
    matrices = [scaler.transform(x.loc[m]) for m in masks]
    classes = sorted(train.target.unique())
    if not set(valid.target).union(test.target).issubset(classes):
        raise ValueError('validation or test contains class absent from training')
    candidates = []
    for c in (.01, .1, 1.):
        model = LogisticRegression(penalty='l1', solver='saga', C=c, max_iter=2000, random_state=42, tol=.001)
        model.fit(matrices[0], train.target)
        score = log_loss(valid.target, model.predict_proba(matrices[1]), labels=model.classes_)
        candidates.append((score, c, model))
    valid_loss, chosen_c, model = min(candidates, key=lambda v: v[0])
    selected = np.any(np.abs(model.coef_) > 1e-8, axis=0)
    rng = np.random.default_rng(42)
    user_ids, inverse = np.unique(train.user_id.to_numpy(), return_inverse=True)
    bootstrap_selection, bootstrap_iterations = [], []
    for _ in range(5):
        sampled = rng.integers(0, len(user_ids), size=len(user_ids))
        weights = np.bincount(sampled, minlength=len(user_ids))[inverse]
        bootstrap_model = LogisticRegression(penalty='l1', solver='saga', C=chosen_c, max_iter=2000, random_state=42, tol=.001)
        bootstrap_model.fit(matrices[0], train.target, sample_weight=weights)
        bootstrap_iterations.append(int(bootstrap_model.n_iter_.max()))
        bootstrap_selection.append(np.any(np.abs(bootstrap_model.coef_) > 1e-8, axis=0))
    frequencies = np.mean(bootstrap_selection, axis=0)
    importance = permutation_importance(model, matrices[1], valid.target, scoring='neg_log_loss',
                                        n_repeats=3, random_state=42, n_jobs=1)
    ranking = sorted(({'feature': name, 'importance': float(imp), 'selected_l1': bool(flag)}
                      for name, imp, flag in zip(x.columns, importance.importances_mean, selected)),
                     key=lambda row: row['importance'], reverse=True)
    result = {'split': bounds, 'rows': {'train': len(train), 'validation': len(valid), 'test': len(test)},
              'users': {name: int(part.user_id.nunique()) for name, part in [('train', train), ('validation', valid), ('test', test)]},
              'target_counts': {name: part.target.value_counts().to_dict() for name, part in [('train', train), ('validation', valid), ('test', test)]},
              'train_missing_fraction': missing.to_dict(), 'transform_selection': transforms, 'constant_train_features': keep.index[~keep].tolist(),
              'selected_features': x.columns[selected].tolist(), 'candidate_count': len(feature_names),
              'bootstrap_selection_frequency': dict(zip(x.columns, frequencies.tolist())),
              'optimization': {'max_iter': 2000, 'selected_iterations': int(model.n_iter_.max()),
                               'bootstrap_iterations': bootstrap_iterations},
              'validation_log_loss': float(valid_loss), 'selected_C': chosen_c,
              'test_log_loss': float(log_loss(test.target, model.predict_proba(matrices[2]), labels=model.classes_)),
              'test_accuracy': float(accuracy_score(test.target, model.predict(matrices[2]))),
              'test_majority_accuracy': float(test.target.eq(train.target.value_counts().index[0]).mean()),
              'validation_permutation_importance': ranking,
              'warning': 'Prediction-only; selected coefficients and importance are not causal effects.'}
    return result, masks


def nts_parent_candidates(summaries, train_users, train_end, max_windows=1500, seed=42):
    """Fit official NTS on independent consecutive-session windows, not a regular clock series."""
    features = ['view', 'cart', 'purchase', 'duration_minutes', 'unique_products', 'unique_categories', 'mean_event_price']
    train = summaries[summaries.user_id.isin(train_users) & (summaries['last'] < train_end)].copy()
    # Missing price is not evidence of a zero price; omit incomplete windows.
    if train.remove_from_cart.nunique() > 1:
        features.append('remove_from_cart')
    grouped = train.groupby(['user_id', 'order_block'], sort=False)
    previous = grouped[features].shift()
    complete = train.safe_start & train.safe_end
    valid = previous.notna().all(axis=1) & train[features].notna().all(axis=1) & complete & grouped.safe_start.shift().eq(True) & grouped.safe_end.shift().eq(True)
    current = train.loc[valid, features].to_numpy(float)
    previous = previous.loc[valid].to_numpy(float)
    if len(current) < 20:
        raise ValueError('not enough consecutive-session windows for NTS')
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(current), min(max_windows, len(current)), replace=False)
    previous, current = previous[idx], current[idx]
    mean = np.vstack((previous, current)).mean(axis=0)
    scale = np.vstack((previous, current)).std(axis=0)
    if (scale == 0).any():
        raise ValueError('constant NTS variable')
    windows = np.stack(((previous - mean) / scale, (current - mean) / scale), axis=2)
    source = PROJECT_ROOT / 'external/NTS-NOTEARS/notears'
    if not (source / 'nts_notears.py').exists():
        raise FileNotFoundError('official NTS-NOTEARS source not found')
    sys.path.insert(0, str(source))
    original_utils = sys.modules.get('utils')
    sys.modules['utils'] = types.ModuleType('utils')
    try:
        official = importlib.import_module('nts_notears')
    finally:
        if original_utils is None:
            del sys.modules['utils']
        else:
            sys.modules['utils'] = original_utils
        sys.path.remove(str(source))
    torch.manual_seed(seed)
    prior = [{'from_node': '*', 'from_lag': 0, 'to_node': '*', 'lower_bound': 0, 'upper_bound': 0}]
    model = official.NTS_NOTEARS(dims=[len(features), 4, 1], bias=True, number_of_lags=1,
                                  prior_knowledge=prior, variable_names_no_time=features).double()
    tensor = torch.tensor(windows, dtype=torch.float64)
    optimizer = official.LBFGSBScipy(model.parameters())
    optimizer.assign_bounds(model)
    def closure():
        optimizer.zero_grad()
        hidden = (model.conv1d_pos(tensor) - model.conv1d_neg(tensor)).squeeze(2).view(-1, len(features), 4)
        for layer in model.fc2:
            hidden = layer(torch.sigmoid(hidden))
        loss = .5 * ((hidden.squeeze(2) - tensor[:, :, -1]) ** 2).sum() / len(tensor)
        loss = loss + .005 * model.fc1_l1_reg() + .005 * model.l2_reg()
        loss.backward()
        return loss
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        optimizer.step(closure)
    finally:
        torch.set_default_dtype(previous_dtype)
    instantaneous, lagged = model.fc1_to_adj()
    edges = [{'source': features[i], 'target': features[j], 'lag': 1, 'weight': float(lagged[i, j])}
             for i in range(len(features)) for j in range(len(features)) if lagged[i, j] >= .3]
    return {'features': features, 'training_windows_available': int(valid.sum()), 'windows_sampled': len(idx),
            'threshold': .3, 'edges': edges, 'instantaneous_max_abs': float(np.abs(instantaneous).max()),
            'assumptions': ['All instantaneous edges forbidden by prior; only lag-1 session edges allowed.',
                            'Official optimizer does not expose convergence status; graph stability was not tested.',
                            'Consecutive observed sessions are irregular in clock time; missing sessions and user heterogeneity are not modeled.',
                            'This is exploratory structure, not identified causal effects or next-action parents.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=('electronics', 'cosmetics'))
    parser.add_argument('--sample-modulus', type=int, default=100)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'artifacts/feature-selection')
    args = parser.parse_args()
    data, audit = load_events(args.source, args.sample_modulus)
    panel, sessions = make_panel(data, data.event_time.max())
    result, masks = select_features(panel)
    result['audit'] = audit
    result['source'] = args.source
    result['nts'] = nts_parent_candidates(sessions, set(panel.loc[masks[0], 'user_id']), pd.Timestamp(result['split']['train_end']))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f'{args.source}.json'
    if output.exists():
        raise FileExistsError(f'output exists: {output}; choose a new --output directory')
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
