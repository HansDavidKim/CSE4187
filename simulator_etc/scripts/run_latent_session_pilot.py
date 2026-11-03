"""Fit/evaluate prospective-prefix session IRL + return hazard on two sources."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits

from simulate.latent_session_model import fit_reward, path_probabilities
from analysis.eda import sessions
from utils.events import load_events

OUT = Path('artifacts/latent-session-pilot')
EDGES = np.array([0., 1., 6., 24., 72., 168., 336., 720.])
COMMON = ['view', 'cart', 'purchase']


def hazard_rows(s, cutoff, features):
    available = (cutoff - s['end']).dt.total_seconds().to_numpy()/3600
    gap = (s['next_start'] - s['end']).dt.total_seconds().to_numpy()/3600
    event = np.isfinite(gap) & (gap <= available)
    elapsed = np.where(event, gap, available)
    x, y = [], []
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        valid = (available >= hi) & (elapsed >= lo)
        if not valid.any():
            continue
        bins = np.zeros((valid.sum(), len(EDGES)-1)); bins[:, i] = 1
        x.append(np.column_stack([bins, features[valid]]))
        y.append((event[valid] & (elapsed[valid] < hi)).astype(int))
    return np.concatenate(x), np.concatenate(y)


def return_probs(model, features, horizon):
    survival = np.ones(len(features))
    for i, hi in enumerate(EDGES[1:]):
        if hi > horizon:
            break
        bins = np.zeros((len(features), len(EDGES)-1)); bins[:, i] = 1
        survival *= 1-model.predict_proba(np.column_stack([bins, features]))[:, 1]
    return 1-survival


def score_paths(paths, probs, observed):
    lookup = {p: i for i, p in enumerate(paths)}
    idx = np.array([lookup[tuple(p)] for p in observed])
    nll = -np.log(np.clip(probs[np.arange(len(idx)), idx], 1e-15, 1)).sum()
    observed_len = np.bincount([len(p) for p in observed], minlength=4)[1:4]/len(observed)
    predicted_len = [float(probs[:, [len(p) == k for p in paths]].sum(axis=1).mean()) for k in (1, 2, 3)]
    return {'paths': len(observed), 'nll_per_decision_including_end': float(nll/sum(len(p)+1 for p in observed)),
            'observed_length_share': observed_len.tolist(), 'predicted_length_share': predicted_len}


def run(source):
    events, quality = load_events(source)
    if source == 'electronics':
        uid = events['user'].drop_duplicates()
        hashed = pd.util.hash_pandas_object(uid, index=False).to_numpy()
        events = events[events.user.isin(uid[hashed % 10 == 0])].copy()
    start, end = events['at'].min().ceil('D'), events['at'].max().floor('D')
    events = events[(events['at'] >= start) & (events['at'] < end)].copy()
    cutoff = start + (end-start)*.70
    events = events.sort_values(['user', 'at'], kind='stable')
    delta = events.groupby('user', sort=False)['at'].diff()
    events['sid'] = (delta.isna() | delta.gt(pd.Timedelta(minutes=30))).groupby(events.user, sort=False).cumsum()
    actions = COMMON + (['remove'] if source == 'cosmetics' else [])
    sequences = events.groupby(['user', 'sid'], sort=False)['action'].agg(list)
    s = sessions(events[['user', 'at', 'action']], end)
    s['rank'] = s.groupby('user', sort=False).cumcount()
    first = s[s['rank'].eq(0)].set_index('user')
    second = s[s['rank'].eq(1)].set_index('user').copy()
    third = s[s['rank'].eq(2)].set_index('user')
    second['third_sid'] = third.sid.reindex(second.index)
    second['third_end'] = third['end'].reindex(second.index)
    for col in ('view', 'cart', 'purchase', 'remove', 'events'):
        second['first_'+col] = first[col].reindex(second.index)
    second = second.reset_index()
    user_hash = pd.util.hash_pandas_object('split:' + second['user'], index=False).to_numpy()
    second['split'] = np.where((user_hash % 5 != 0) & (second['end'] + pd.Timedelta(days=30) <= cutoff), 'train',
                       np.where((user_hash % 5 == 0) & (second['end'] >= cutoff) &
                                (second['end'] + pd.Timedelta(days=30) <= end), 'test', 'excluded'))
    prefix_cols = ['first_view', 'first_cart', 'first_purchase', 'first_remove', 'first_events',
                   'view', 'cart', 'purchase', 'remove', 'events', 'previous_gap_hours']
    tr = second[second.split.eq('train')].copy()
    te = second[second.split.eq('test')].copy()
    if min(len(tr), len(te)) < 30:
        raise ValueError(f'insufficient landmarks: train={len(tr)}, test={len(te)}')
    scaler = StandardScaler().fit(tr[prefix_cols])
    with threadpool_limits(limits=1):
        gmm = GaussianMixture(n_components=3, covariance_type='diag', reg_covar=.05,
                              random_state=42, n_init=3).fit(scaler.transform(tr[prefix_cols]))
    def latent(frame):
        return gmm.predict_proba(scaler.transform(frame[prefix_cols]))
    ztr, zte = latent(tr), latent(te)
    # The first two mixture probabilities are used; third is reference to avoid collinearity.
    def hazard_features(frame, z):
        return np.column_stack([frame[['view', 'cart', 'purchase', 'remove', 'events', 'previous_gap_hours']].to_numpy(), z[:, :2]])
    htr, hte = hazard_features(tr, ztr), hazard_features(te, zte)
    x, y = hazard_rows(tr, cutoff, htr)
    hazard = LogisticRegression(max_iter=1000, random_state=42).fit(x, y)
    xb, yb = hazard_rows(tr, cutoff, np.empty((len(tr), 0)))
    baseline = LogisticRegression(max_iter=1000, random_state=42).fit(xb, yb)
    hazard_scores = {}
    for h in (24, 168, 720):
        truth = ((te.next_start-te.end).dt.total_seconds().to_numpy()/3600 <= h).astype(int)
        p = return_probs(hazard, hte, h)
        pb = return_probs(baseline, np.empty((len(te), 0)), h)
        hazard_scores[str(h)] = {'n': len(te), 'observed_return_rate': float(truth.mean()),
                                 'predicted_rate': float(p.mean()), 'brier': float(brier_score_loss(truth, p)),
                                 'baseline_brier': float(brier_score_loss(truth, pb))}

    def trajectories(frame, z, stop):
        valid = frame.third_sid.notna() & frame.third_end.notna() & frame.third_end.le(stop) & \
                frame.next_start.le(frame.end + pd.Timedelta(days=30))
        indices = np.flatnonzero(valid.to_numpy())
        paths, lats = [], []
        for i in indices:
            row = frame.iloc[i]
            seq = sequences.loc[(row.user, int(row.third_sid))]
            if 1 <= len(seq) <= 3:
                paths.append(tuple(actions.index(a) for a in seq))
                lats.append(z[i, :2])
        return paths, np.asarray(lats).reshape(-1, 2), len(indices)
    train_paths, train_z, train_returned = trajectories(tr, ztr, cutoff)
    test_paths, test_z, test_returned = trajectories(te, zte, end)
    if min(len(train_paths), len(test_paths)) < 10:
        raise ValueError(f'insufficient short sessions {len(train_paths)}, {len(test_paths)}')
    with threadpool_limits(limits=1):
        reward, optimization = fit_reward(train_z, train_paths, len(actions), l1=.1)
        base_reward, base_optimization = fit_reward(np.empty((len(train_paths), 0)), train_paths, len(actions), l1=.1)
    paths, p = path_probabilities(test_z, reward, len(actions))
    _, pbase = path_probabilities(np.empty((len(test_paths), 0)), base_reward, len(actions))
    result = {'source': source, 'quality': quality, 'actions': actions, 'start_utc': start.isoformat(),
              'train_cutoff_utc': cutoff.isoformat(), 'end_utc': end.isoformat(),
              'sample': 'pandas user hash modulo 10' if source == 'electronics' else 'user_id modulo 100',
              'landmarks': {'train': len(tr), 'test': len(te)}, 'gmm_components': 3,
              'gmm_train_component_weights': gmm.weights_.tolist(),
              'gmm_test_mean_responsibilities': zte.mean(axis=0).tolist(),
              'hazard_training_rows': len(y), 'hazard_training_events': int(y.sum()), 'return_metrics': hazard_scores,
              'irl': {'train_selected_short_sessions': len(train_paths), 'test_selected_short_sessions': len(test_paths),
                      'train_returned_sessions_all_lengths': train_returned, 'test_returned_sessions_all_lengths': test_returned,
                      'max_events': 3, 'temperature': 1, 'l1': .1, 'optimization': optimization,
                      'baseline_optimization': base_optimization, 'test': score_paths(paths, p, test_paths),
                      'test_no_latent': score_paths(paths, pbase, test_paths), 'reward_coefficients': reward.tolist()},
              'limitations': ['First two observed sessions are selection landmarks, not necessarily new user histories.',
                              '30min timeout sessionization assumes missing events do not break sessions and inferred END is not observed logout.',
                              'Next session observed within 30d and at most 3 events required for IRL; long sessions excluded.',
                              'Forced END at horizon 3; truncation is not a natural termination model.',
                              'Sequence paths and timeouts are observational; no coupon treatment or causal uplift is identified.',
                              'Model gives no gap beyond 30 days and no state-dependent intervention effects.',
                              'GMM on two-session features, not learned user-wise causal utility; clusters are source-specific.',
                              'User hash split and single temporal split; logging completeness, exposure, item feasibility and ordering ties unverified.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{source}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    joblib.dump({'actions': actions, 'prefix_cols': prefix_cols, 'scaler': scaler, 'gmm': gmm,
                 'reward': reward, 'hazard': hazard, 'hazard_edges_hours': EDGES}, OUT / f'{source}.joblib')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    for source in sys.argv[1:]:
        if source not in ('electronics', 'cosmetics'):
            raise ValueError(source)
        run(source)
