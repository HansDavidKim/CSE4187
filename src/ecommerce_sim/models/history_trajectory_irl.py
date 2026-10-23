"""Finite-horizon MaxEnt behavior model with strictly preceding history features."""
from itertools import combinations, product

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ACTIONS = ('view', 'cart', 'purchase')
STATE_NAMES = ('intercept', 'session_views', 'session_carts', 'session_purchases',
               'previous_sessions', 'previous_purchases')
PATHS = np.array(list(product(range(3), repeat=3)), dtype=np.int8)
BASE_NAMES = STATE_NAMES[1:]
QUADRATIC_NAMES = (STATE_NAMES + tuple(f'{name}^2' for name in BASE_NAMES) +
                   tuple(f'{a}*{b}' for a, b in combinations(BASE_NAMES, 2)))
LAG_NAMES = tuple(f'lag{lag}_{action}' for lag in (1, 2) for action in ACTIONS)


def feature_counts(initial, paths=PATHS, quadratic=False, lags=False):
    """Return cumulative action-specific features for each candidate path.

    initial is [session view/cart/purchase, completed sessions, earlier purchases].
    View is the reference action. Fixed log scaling needs no test-fitted statistics.
    """
    initial = np.asarray(initial, dtype=float)
    paths = np.asarray(paths, dtype=int)
    if initial.ndim != 2 or initial.shape[1] != (7 if lags else 5) or np.any(initial[:, :5] < 0):
        raise ValueError('initial history must be nonnegative [n, 5]')
    if lags and (np.any((initial[:, 5:] < -1) | (initial[:, 5:] > 2)) or
                 np.any(initial[:, 5:] != np.floor(initial[:, 5:]))):
        raise ValueError('lag actions must be -1 (unavailable) or 0..2')
    if paths.ndim != 2 or np.any((paths < 0) | (paths > 2)):
        raise ValueError('paths must be a matrix of common action IDs')
    n, m = len(initial), len(paths)
    counts = np.broadcast_to(initial[:, None, :5], (n, m, 5)).copy()
    if lags:
        lag1 = np.broadcast_to(initial[:, None, 5], (n, m)).copy()
        lag2 = np.broadcast_to(initial[:, None, 6], (n, m)).copy()
    term_count = len(QUADRATIC_NAMES) if quadratic else len(STATE_NAMES)
    term_count += len(LAG_NAMES) if lags else 0
    result = np.zeros((n, m, 2, term_count))
    for action in paths.T:
        x = np.concatenate((np.ones((n, m, 1)), np.log1p(counts) / np.log(65)), axis=2)
        if quadratic:
            z = x[:, :, 1:]
            x = np.concatenate((x, z * z,
                                np.stack([z[:, :, i] * z[:, :, j]
                                          for i, j in combinations(range(5), 2)], axis=2)), axis=2)
        if lags:
            x = np.concatenate((x, np.stack([lag == a for lag in (lag1, lag2)
                                             for a in range(3)], axis=2)), axis=2)
        for a in (1, 2):
            result[:, :, a - 1, :] += x * (action == a)[None, :, None]
        for a in range(3):
            counts[:, :, a] += (action == a)[None, :]
        if lags:
            lag2 = lag1
            lag1 = np.broadcast_to(action[None, :], (n, m))
    return result.reshape(n, m, -1)


def fit_maxent(initial, observed, penalty=1.0, quadratic=False, l1=0.0, lags=False):
    observed = np.asarray(observed, dtype=int)
    if l1 < 0 or penalty < 0:
        raise ValueError('regularization strengths must be nonnegative')
    if observed.shape != (len(initial), 3) or not len(initial):
        raise ValueError('expected nonempty three-action trajectories')
    candidates = feature_counts(initial, quadratic=quadratic, lags=lags)
    index = observed[:, 0] * 9 + observed[:, 1] * 3 + observed[:, 2]
    if np.any((observed < 0) | (observed > 2)):
        raise ValueError('unexpected action')
    empirical = candidates[np.arange(len(initial)), index]

    def objective(theta):
        scores = candidates @ theta
        log_z = logsumexp(scores, axis=1)
        probs = np.exp(scores - log_z[:, None])
        loss = np.sum(log_z - np.sum(empirical * theta, axis=1)) + penalty * theta @ theta / 2
        gradient = np.einsum('nm,nmk->k', probs, candidates) - empirical.sum(axis=0) + penalty * theta
        return loss, gradient

    dimension = candidates.shape[-1]
    if l1:
        def split_objective(split):
            loss, gradient = objective(split[:dimension] - split[dimension:])
            return loss + l1 * split.sum(), np.r_[gradient + l1, -gradient + l1]
        result = minimize(split_objective, np.zeros(2 * dimension), jac=True, method='L-BFGS-B',
                          bounds=[(0, None)] * (2 * dimension),
                          options={'maxiter': 1500, 'ftol': 1e-11, 'gtol': 1e-5})
        theta = result.x[:dimension] - result.x[dimension:]
    else:
        result = minimize(objective, np.zeros(dimension), jac=True, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-10})
        theta = result.x
    if not result.success:
        raise RuntimeError(result.message)
    return theta, {'iterations': int(result.nit), 'objective': float(result.fun)}


def evaluate(initial, observed, theta, quadratic=False, lags=False):
    candidates = feature_counts(initial, quadratic=quadratic, lags=lags)
    scores = candidates @ theta
    log_probs = scores - logsumexp(scores, axis=1)[:, None]
    index = observed[:, 0] * 9 + observed[:, 1] * 3 + observed[:, 2]
    probabilities = np.exp(log_probs)
    first = np.stack([probabilities[:, PATHS[:, 0] == a].sum(axis=1) for a in range(3)], axis=1)
    return {'trajectory_nll_per_action': float(-log_probs[np.arange(len(index)), index].mean() / 3),
            'first_action_nll': float(-np.log(first[np.arange(len(index)), observed[:, 0]]).mean()),
            'first_action_observed_share': np.bincount(observed[:, 0], minlength=3).astype(float).__truediv__(len(index)).tolist(),
            'first_action_predicted_share': first.mean(axis=0).tolist()}
