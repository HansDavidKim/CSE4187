"""User-specific contextual maximum-entropy reward and choice models."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression


def weekly_choices(values: np.ndarray, features: list[str]) -> np.ndarray:
    purchase = values[..., features.index('transactions')] > 0
    coupon = values[..., features.index('coupon_discount')] > 0
    if np.any(coupon & ~purchase):
        raise ValueError('coupon discount without a recorded purchase')
    return np.where(~purchase, 0, np.where(coupon, 2, 1)).astype(int)


def state_spec(dag: dict) -> list[tuple[str, int]]:
    if dag.get('converged') is not True or dag.get('instantaneous_is_dag') is not True:
        raise ValueError('NTS graph must be converged and acyclic')
    targets = {'transactions', 'quantity', 'coupon_discount', 'offers_started', 'active_offers'}
    nodes = {(str(e['source']), int(e['lag'])) for e in dag['edges']
             if str(e['target']) in targets and int(e['lag']) > 0}
    # Explicit controls, not claimed to be discovered parents of purchase.
    nodes |= {('transactions', 1), ('active_offers', 1)}
    return sorted(nodes, key=lambda x: (x[1], x[0]))


def make_states(values: np.ndarray, features: list[str], spec: list[tuple[str, int]], times: np.ndarray) -> np.ndarray:
    if any(lag < 1 or min(times) < lag for _, lag in spec):
        raise ValueError('all state features must precede the decision')
    return np.stack([values[:, times - lag, features.index(name)] for name, lag in spec], axis=-1)


class UserMaxEnt:
    """Hierarchically regularized one-step MaxEnt IRL; action 0 fixes reward scale origin."""
    def __init__(self, penalty: float = 10.0):
        self.penalty = penalty

    def fit(self, state: np.ndarray, action: np.ndarray) -> 'UserMaxEnt':
        if state.ndim != 3 or action.shape != state.shape[:2] or self.penalty <= 0:
            raise ValueError('expected [users, time, state], aligned actions and positive penalty')
        self.mean_ = state.mean(axis=(0, 1))
        self.scale_ = state.std(axis=(0, 1))
        self.scale_[self.scale_ == 0] = 1
        x = self._design(state)
        n, t, p = x.shape
        if set(np.unique(action)) != {0, 1, 2}:
            raise ValueError('training data must contain all three actions')
        pooled = LogisticRegression(C=1.0, max_iter=500).fit(x.reshape(-1, p)[:, 1:], action.ravel())
        w = np.column_stack((pooled.intercept_, pooled.coef_))
        center = w[1:] - w[0]
        initial = np.broadcast_to(center, (n, 2, p)).copy()

        def objective(flat):
            weights = flat.reshape(n, 2, p)
            logits = np.concatenate((np.zeros((n, t, 1)), np.einsum('ntp,nkp->ntk', x, weights)), axis=2)
            loss = (logsumexp(logits, axis=2) - np.take_along_axis(logits, action[..., None], axis=2)[..., 0]).sum()
            diff = weights - center
            loss += self.penalty * np.square(diff).sum() / 2
            residual = softmax(logits, axis=2)[..., 1:] - (action[..., None] == np.array([1, 2]))
            grad = np.einsum('ntk,ntp->nkp', residual, x) + self.penalty * diff
            return float(loss), grad.ravel()

        result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
                          options={'maxiter': 250, 'ftol': 1e-9})
        if not result.success:
            raise RuntimeError(f'user-wise reward optimization failed: {result.message}')
        self.weights_ = result.x.reshape(n, 2, p)
        self.pooled_weights_ = center
        self.optimization_iterations_ = result.nit
        return self

    def _design(self, states):
        standardized = (states - self.mean_) / self.scale_
        return np.concatenate((np.ones((*standardized.shape[:-1], 1)), standardized), axis=-1)

    def utilities(self, states: np.ndarray, *, pooled: bool = False) -> np.ndarray:
        x = self._design(states)
        weights = np.broadcast_to(self.pooled_weights_, (len(x), *self.pooled_weights_.shape)) if pooled else self.weights_
        if len(x) != len(weights):
            raise ValueError('state users do not match fitted users')
        utility = np.einsum('ntp,nkp->ntk', x, weights)
        return np.concatenate((np.zeros((*utility.shape[:-1], 1)), utility), axis=-1)
