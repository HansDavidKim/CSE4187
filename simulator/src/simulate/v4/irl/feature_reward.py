"""K=1 linear-feature reward fitted through the existing soft Bellman policy.

The caller supplies a frozen state basis and training counts/dynamics. Feature
scaling, state construction and strict-past availability must be fitted and
checked by the caller; this module cannot establish those properties.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from . import fit_latent_shopping as shopping


def _number(value, name, lower, upper=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < lower
            or upper is not None and value >= upper):
        raise ValueError(f"invalid {name}")
    return float(value)


def _inputs(basis, counts, matrices, mask, discount, regularization):
    basis, counts = np.asarray(basis, dtype=float), np.asarray(counts, dtype=float)
    mask = np.asarray(mask)
    if (basis.ndim != 2 or min(basis.shape) < 1 or not np.isfinite(basis).all()
            or counts.shape != (len(basis), 5) or not np.isfinite(counts).all()
            or (counts < 0).any() or not math.isfinite(float(counts.sum())) or counts.sum() <= 0
            or mask.shape != counts.shape or mask.dtype.kind != "b"
            or not mask.any(axis=1).all() or counts[~mask].any()):
        raise ValueError("finite basis, positive supported counts and boolean action mask required")
    discount = _number(discount, "discount", 0, 1)
    regularization = _number(regularization, "regularization", 0)
    if len(matrices) != 5:
        raise ValueError("five sparse transition matrices required")
    transitions = []
    for action, matrix in enumerate(matrices):
        if not sparse.issparse(matrix):
            raise ValueError("transition matrices must be sparse")
        matrix = sparse.csr_matrix(matrix, dtype=float, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if (matrix.shape != (len(basis), len(basis)) or not np.isfinite(matrix.data).all()
                or (matrix.data < 0).any()):
            raise ValueError("transition shape or probabilities invalid")
        mass = np.asarray(matrix.sum(axis=1)).ravel()
        if action == 4:
            if matrix.nnz:
                raise ValueError("END must have zero continuation transitions")
        elif (not np.allclose(mass[mask[:, action]], 1, atol=1e-10, rtol=0)
              or not (np.isclose(mass, 0, atol=1e-10, rtol=0)
                      | np.isclose(mass, 1, atol=1e-10, rtol=0)).all()):
            raise ValueError("supported non-END transition rows must sum to one")
        transitions.append(matrix)
    return basis, counts, transitions, mask, discount, regularization


def _offset(value, mask):
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != mask.shape or not np.isfinite(value).all():
        raise ValueError("fixed reward offset must be finite state-by-five")
    return value


def _evaluate(flat, basis, counts, matrices, mask, discount, regularization, prior=None,
              reward_offset=None):
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (4 * basis.shape[1],) or not np.isfinite(flat).all():
        raise ValueError("finite flattened4-by-feature reward weights required")
    theta = flat.reshape(4, basis.shape[1])
    difference = theta if prior is None else theta - prior
    reward = np.column_stack([basis @ theta.T, np.zeros(len(basis))])
    if reward_offset is not None:
        reward += reward_offset
    if not np.isfinite(reward).all():
        raise ValueError("feature reward overflowed")
    ll, gradient, statistics = shopping.policy_likelihood_gradient(
        reward, counts, matrices, mask, discount)
    total = float(counts.sum())
    loss = -ll / total + regularization * np.square(difference).sum() / 2
    derivative = -(gradient[:, :4].T @ basis) / total + regularization * difference
    if not math.isfinite(loss) or not np.isfinite(derivative).all():
        raise ValueError("nonfinite feature reward objective or gradient")
    return float(loss), derivative.ravel(), float(-ll / total), statistics


def feature_reward_objective(flat, basis, counts, matrices, mask, *, discount=.95,
                             regularization=.001, prior=None, reward_offset=None):
    """Return penalized mean negative log likelihood and its exact flat gradient."""
    inputs = _inputs(basis, counts, matrices, mask, discount, regularization)
    prior = _prior(prior, inputs[0].shape[1])
    offset = _offset(reward_offset, inputs[3])
    loss, gradient, _, _ = _evaluate(flat, *inputs, prior, offset)
    return loss, gradient


def _prior(value, width):
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != (4, width) or not np.isfinite(value).all():
        raise ValueError("prior weights must be finite4-by-feature")
    return value


def fit_feature_reward(basis, counts, matrices, mask, *, feature_names, discount=.95,
                       regularization=.001, max_iter=1000, initial=None, prior=None,
                       reward_offset=None):
    """Fit four linear reward vectors; learned END reward and continuation are zero.

    ``initial`` may be4-by-p or flattened4p. The actual accepted-iterate trace
    includes iteration0. Reaching a budget is reported as nonconvergence.
    A fixed offset (including END) can supply a reference policy's log probability.
    """
    inputs = _inputs(basis, counts, matrices, mask, discount, regularization)
    basis, counts, matrices, mask, discount, regularization = inputs
    prior = _prior(prior, basis.shape[1])
    offset = _offset(reward_offset, mask)
    names = list(feature_names)
    if (len(names) != basis.shape[1] or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("unique feature names must match basis columns")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    start = np.zeros((4, basis.shape[1])) if initial is None else np.asarray(initial, dtype=float)
    if start.shape not in ((4, basis.shape[1]), (4 * basis.shape[1],)) or not np.isfinite(start).all():
        raise ValueError("initial weights must be finite4-by-p or flattened4p")
    cached, evaluations = {}, 0

    def objective(flat):
        nonlocal evaluations
        if "x" not in cached or not np.array_equal(flat, cached["x"]):
            cached["x"] = flat.copy()
            cached["result"] = _evaluate(flat, *inputs, prior, offset)
            evaluations += 1
        return cached["result"][:2]

    def snapshot(flat, iteration):
        loss, gradient = objective(flat)
        return {"iteration": int(iteration), "objective": loss, "raw_nll": cached["result"][2],
                "gradient_max_abs": float(np.abs(gradient).max()),
                "bellman_residual": float(cached["result"][3][2])}

    trace = [snapshot(start.ravel(), 0)]
    optimum = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B",
        callback=lambda flat: trace.append(snapshot(flat, len(trace))),
        options={"maxiter": int(max_iter), "gtol": 1e-6, "ftol": 1e-11})
    final = snapshot(optimum.x, optimum.nit)
    if trace[-1] != final:
        trace.append(final)
    theta = optimum.x.reshape(4, basis.shape[1])
    probability, value, _ = cached["result"][3]
    result = {"feature_names": names, "discount": discount, "regularization": regularization,
        "reward_weights": theta.tolist(), "common_reward": (basis @ theta.T).tolist(),
        "policy": probability.tolist(), "value": value.tolist(),
        "fit": {"converged": bool(optimum.success), "message": str(optimum.message),
            "iterations": int(optimum.nit), "gradient_max_abs": final["gradient_max_abs"],
            "bellman_residual": final["bellman_residual"], "raw_nll": final["raw_nll"],
            "objective": final["objective"], "objective_evaluations": evaluations,
            "observations": float(counts.sum()), "parameter_count": int(theta.size),
            "objective_definition": "-conditional soft-Bellman log likelihood/count total + "
                                    + ("L2(theta)/2" if prior is None else "L2(theta-prior)/2"),
            "trace": trace}}
    if offset is not None:
        result["fixed_reward_offset"] = offset.tolist()
        result["fit"]["objective_definition"] += "; fixed action reward offset included in soft Bellman solve"
    return result
