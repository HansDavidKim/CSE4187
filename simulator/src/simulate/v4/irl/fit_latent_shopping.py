"""Observed-history sequential IRL with a persistent, source-fitted customer mixture.

Rewards are learned through the soft Bellman policy likelihood, not a local
classifier. Anonymous component offsets are private simulator traits. Their
default posterior bank uses training-prefix demonstrations; an explicitly
marked reconstruction can supply source-observation traits for synthesis only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse.linalg import spsolve
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

from simulate.v4.irl import fit_causal_shopping as old
from simulate.v4.irl.shopping_history import ShoppingHistory, SHOPPING_HISTORY_FEATURES


ACTIONS = old.ACTIONS
DISCOUNT = .95
PENALTIES = (1e-5, 1e-4)
TYPE = "observed_history_persistent_mixture_soft_bellman_irl"
PROGRESS_EDGES = (0, 1, 2, 3, 4, 5, 10, 20, 50)
FEATURE_NAMES = [*old.FEATURES, "log_previous_action_run_length",
    *[f"progress_bucket_{i}" for i in range(10)], *[f"cart_quantity_bucket_{i}" for i in range(4)],
    *[f"history_{i}_{part}" for i in range(5) for part in ("value", "observed")]]
CONTEXT_KEYS = [*old.CONTEXT_KEYS, "previous_action_run_length", *SHOPPING_HISTORY_FEATURES]
_RUNTIME = {}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cart_bucket(quantity):
    return 0 if quantity == 0 else 1 if quantity == 1 else 2 if quantity <= 4 else 3


def context_vector(context):
    core = old.context_vector(context)
    run = float(context.get("previous_action_run_length", 0))
    progress = float(context.get("session_decision_count", 0))
    if not np.isfinite(run) or run < 0 or not run.is_integer() or (progress == 0) != (run == 0):
        raise ValueError("previous_action_run_length must describe the observed current session")
    bucket = int(np.searchsorted(PROGRESS_EDGES, progress, side="left"))
    quantity = float(context.get("cart_total_quantity", 0))
    history = []
    for index, name in enumerate(SHOPPING_HISTORY_FEATURES):
        value = context.get(name)
        missing = value is None or pd.isna(value)
        if missing:
            history.extend([0.0, 0.0])
            continue
        number = float(value)
        if not np.isfinite(number) or number < 0 or index == 3 and number > 1:
            raise ValueError(f"invalid observed completed-history feature: {name}")
        history.extend([number if index == 3 else np.log1p(number), 1.0])
    return np.concatenate([core, [np.log1p(run)], np.eye(10)[bucket], np.eye(4)[cart_bucket(quantity)], history])


def build_source(rows, observation_end, seed=42):
    """Augment the immutable legacy row replay with conservative completed history."""
    data = old.build_demonstrations(rows, observation_end, seed)
    source = rows.sort_values(["user_id", "at", "source_event_id"], kind="stable")
    vectors, usable, units = [], [], []
    cursor = 0
    end = pd.Timestamp(observation_end)
    audit = Counter()
    for uid, user in source.groupby("user_id", sort=True):
        history, cart = ShoppingHistory(), Counter()
        for sid, session in user.groupby("session", sort=False):
            sequence = list(session.itertuples(index=False))
            history.begin_session(sid, sequence[0].at)
            at, previous, run, step = sequence[0].at, "start", 0, 0
            index = 0
            while index < len(sequence):
                event = sequence[index]
                stop = index + 1
                if event.action == "purchase":
                    while stop < len(sequence) and sequence[stop].action == "purchase" and sequence[stop].at == event.at:
                        stop += 1
                record = data["records"].iloc[cursor]
                assert record.user_id == str(uid) and record.session == sid and record.step == step and record["at"] == at
                # Reconstruct the same core context from its invertible legacy feature vector.
                x = data["features"][cursor]
                context = {"previous_action": previous, "cart_total_quantity": sum(cart.values()),
                    "unit_price_minor": np.expm1(x[1]) if x[2] else None,
                    "days_since_first_observed": np.expm1(x[3]),
                    "days_since_purchase": np.expm1(x[4]) if x[5] else None,
                    "purchase_count": np.expm1(x[6]), "session_decision_count": step,
                    "prior_gap_seconds": np.expm1(x[8]), "purchase_in_current_session": bool(x[9]),
                    "previous_action_run_length": run, **history.snapshot(at)}
                vectors.append(context_vector(context))
                matched = event.action != "remove" or cart[event.product_id] > 0
                usable.append(matched)
                units.append(stop - index if event.action == "purchase" else 1)
                if not matched:
                    audit["unmatched_remove_transition_pairs_excluded"] += 1
                for item in sequence[index:stop]:
                    if item.action == "cart":
                        cart[item.product_id] += 1
                    elif item.action in {"remove", "purchase"} and cart[item.product_id]:
                        cart[item.product_id] -= 1
                history.record_action(sid, event.at, event.action, quantity=stop - index,
                    view_price_minor=(None if pd.isna(event.price_minor) else float(event.price_minor)) if event.action == "view" else None)
                run = run + 1 if event.action == previous else 1
                previous, at = event.action, sequence[stop - 1].at
                step += 1
                cursor += 1
                index = stop
            record = data["records"].iloc[cursor]
            assert record.action == 4 and record["at"] == at
            x = data["features"][cursor]
            context = {"previous_action": previous, "cart_total_quantity": sum(cart.values()),
                "unit_price_minor": np.expm1(x[1]) if x[2] else None,
                "days_since_first_observed": np.expm1(x[3]),
                "days_since_purchase": np.expm1(x[4]) if x[5] else None,
                "purchase_count": np.expm1(x[6]), "session_decision_count": step,
                "prior_gap_seconds": np.expm1(x[8]), "purchase_in_current_session": bool(x[9]),
                "previous_action_run_length": run, **history.snapshot(at)}
            vectors.append(context_vector(context));usable.append(False);units.append(0)
            history.end_session(sid, at, completed=at + pd.Timedelta(minutes=30) <= end)
            cursor += 1
    assert cursor == len(data["records"])
    data["features"] = np.asarray(vectors)
    data["dynamics_usable"] = np.asarray(usable, dtype=bool)
    data["records"]["purchase_item_units"] = units
    data["audit"].update(audit)
    data["audit"]["completed_history_available_rule"] = "last_action_at+30m strictly before decision clock; unsupported fitting sessions remain observed history"
    return data


def select_panel(data, mask):
    panel = old._selected(data, mask, "pooled-session")
    sessions = data["sessions"].loc[mask]
    indices = np.concatenate([np.arange(r.record_start, r.record_end) for r in sessions.itertuples()])
    panel["dynamics_usable"] = data["dynamics_usable"][indices]
    return panel


def _hard_group(features):
    quantities = np.rint(np.expm1(features[:, 0])).astype(int)
    bucket = np.where(quantities == 0, 0, np.where(quantities == 1, 1, np.where(quantities <= 4, 2, 3)))
    return bucket * 2 + (features[:, 7] == 0).astype(int)


def _tree_apply(tree, features):
    features = np.asarray(features, dtype=np.float32)
    left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
    feature, threshold = np.asarray(tree["feature"]), np.asarray(tree["threshold"])
    nodes = np.zeros(len(features), dtype=int)
    active = left[nodes] >= 0
    while active.any():
        indices = np.flatnonzero(active);current = nodes[indices]
        nodes[indices] = np.where(features[indices, feature[current]] <= threshold[current], left[current], right[current])
        active = left[nodes] >= 0
    return nodes


def fit_encoder(panel, max_leaves=64, minimum_leaf=32):
    x, y = panel["features"], panel["records"].action.to_numpy()
    estimator = DecisionTreeClassifier(max_leaf_nodes=max_leaves, min_samples_leaf=minimum_leaf, random_state=42)
    estimator.fit(x, y)
    tree = {"left": estimator.tree_.children_left.tolist(), "right": estimator.tree_.children_right.tolist(),
            "feature": estimator.tree_.feature.tolist(), "threshold": estimator.tree_.threshold.tolist()}
    leaves = _tree_apply(tree, x)
    keys = np.column_stack([leaves, _hard_group(x)])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    means = np.asarray([x[inverse == i].mean(axis=0) for i in range(len(unique))])
    scale = x.std(axis=0);scale[scale < 1e-12] = 1
    return {"schema_version": 1, "feature_names": FEATURE_NAMES, "tree": tree, "state_keys": unique.tolist(),
        "context_means": means.tolist(), "feature_scale": scale.tolist(), "max_tree_leaves": max_leaves,
        "minimum_leaf_records": minimum_leaf, "hard_group": "cart quantity 0/1/2-4/5+ crossed with START/nonSTART",
        "progress_buckets": [0, 1, 2, 3, 4, 5, "6-10", "11-20", "21-50", ">50"],
        "training_only": True, "maximum_states": max_leaves * 8}


def encode(encoder, features):
    x = np.asarray(features)
    leaves = _tree_apply(encoder["tree"], x)
    groups = _hard_group(x)
    keys = np.asarray(encoder["state_keys"]);mapping = {tuple(k): i for i, k in enumerate(keys)}
    means, scale = np.asarray(encoder["context_means"]), np.asarray(encoder["feature_scale"])
    state, fallback = np.empty(len(x), dtype=int), np.zeros(len(x), dtype=bool)
    for i, key in enumerate(zip(leaves, groups)):
        found = mapping.get(key)
        if found is None:
            fallback[i] = True
            candidates = np.flatnonzero(keys[:, 1] == key[1])
            if not len(candidates):
                raise ValueError("no observed source state supports this cart/START group")
            found = int(candidates[np.argmin(((means[candidates] - x[i]) / scale) ** 2 @ np.ones(x.shape[1]))])
        state[i] = found
    return state, fallback


def action_mask(encoder):
    groups = np.asarray(encoder["state_keys"])[:, 1]
    mask = np.ones((len(groups), 5), dtype=bool)
    mask[groups // 2 == 0, 2] = False
    mask[groups % 2 == 1, 4] = False
    return mask


def fit_dynamics(encoder, panel):
    state, fallback = encode(encoder, panel["features"])
    assert not fallback.any()
    y = panel["records"].action.to_numpy();n = len(encoder["state_keys"])
    mask = action_mask(encoder)
    if not mask[state, y].all():
        raise ValueError("unsupported demonstration action")
    matrices, audit = [], {}
    groups = np.asarray(encoder["state_keys"])[:, 1]
    for a in range(5):
        if a == 4:
            matrices.append(sparse.csr_matrix((n, n)));continue
        indices = np.flatnonzero((y == a) & panel["dynamics_usable"])
        if any(i + 1 >= len(y) or panel["starts"][i + 1] for i in indices):
            raise ValueError("missing actual next state")
        counts = sparse.coo_matrix((np.ones(len(indices)), (state[indices], state[indices + 1])), shape=(n, n)).tocsr()
        pooled = np.asarray(counts.sum(axis=0)).ravel()
        if not pooled.sum():
            raise ValueError(f"source has no identified transitions for {ACTIONS[a]}")
        pools = {g: np.asarray(counts[groups == g].sum(axis=0)).ravel() for g in np.unique(groups)}
        rows = []
        scopes = Counter()
        for s in range(n):
            if not mask[s, a]:
                rows.append(sparse.csr_matrix((1, n)));continue
            row = counts.getrow(s)
            if row.sum():
                rows.append(row / row.sum());scopes["state_action"] += 1
            else:
                values = pools[groups[s]]
                scope = "hard_group_action"
                if not values.sum():values, scope = pooled, "pooled_identified_action"
                rows.append(sparse.csr_matrix(values / values.sum()));scopes[scope] += 1
        matrices.append(sparse.vstack(rows, format="csr"))
        audit[ACTIONS[a]] = {"action_records": int((y == a).sum()), "identified_transition_pairs": len(indices),
            "excluded_transition_pairs": int((y == a).sum()) - len(indices), "backoff_rows": dict(scopes)}
    return matrices, mask, state, audit


def soft_policy(reward, matrices, mask, discount=DISCOUNT):
    value = np.zeros(len(reward))
    for iteration in range(5000):
        q = reward + discount * np.column_stack([p @ value for p in matrices])
        q = np.where(mask, q, -np.inf)
        updated = logsumexp(q, axis=1)
        if np.max(np.abs(updated - value)) < 1e-10:
            value = updated;break
        value = updated
    else:raise ValueError("latent soft Bellman iteration did not converge")
    q = np.where(mask, reward + discount * np.column_stack([p @ value for p in matrices]), -np.inf)
    logp = q - logsumexp(q, axis=1, keepdims=True)
    probability = np.exp(logp)
    return probability, value, float(np.max(np.abs(logsumexp(q, axis=1) - value)))


def policy_likelihood_gradient(reward, counts, matrices, mask, discount=DISCOUNT):
    """Exact adjoint derivative through soft Bellman, for arbitrary observed counts."""
    probability, value, residual = soft_policy(reward, matrices, mask, discount)
    logp = np.zeros_like(probability)
    logp[mask] = np.log(np.maximum(probability[mask], 1e-300))
    likelihood = float((counts * logp).sum())
    continuation = sum(sparse.diags(probability[:, a]) @ matrices[a] for a in range(5))
    forcing = discount * sum(matrices[a].T @ counts[:, a] for a in range(5)) - counts.sum(axis=1)
    adjoint = spsolve(sparse.eye(len(reward), format="csc") - discount * continuation.T.tocsc(), forcing)
    gradient = counts + probability * adjoint[:, None]
    return likelihood, gradient, (probability, value, residual)


def customer_counts(panel, state, n_states):
    users, user_index = np.unique(panel["records"].user_id.to_numpy(), return_inverse=True)
    action = panel["records"].action.to_numpy()
    counts = sparse.coo_matrix((np.ones(len(state)), (user_index, state * 5 + action)), shape=(len(users), n_states * 5)).tocsr()
    return users, counts


def empirical_reward(counts, matrices, mask):
    probability = counts.astype(float).copy()
    probability[~mask] = 0
    probability /= probability.sum(axis=1, keepdims=True)
    probability[mask] = np.maximum(probability[mask], 1e-8)
    probability /= probability.sum(axis=1, keepdims=True)
    value = np.where(mask[:, 4], -np.log(np.maximum(probability[:, 4], 1e-300)), 0)
    reward = np.zeros_like(probability)
    for a in range(4):reward[:, a] = np.log(np.maximum(probability[:, a], 1e-300)) + value - DISCOUNT * (matrices[a] @ value)
    reward[~mask] = 0
    return reward[:, :4]


def _reward(base, delta=None, sign=0):
    values = base if delta is None else base + sign * delta[None, :]
    return np.column_stack([values, np.zeros(len(base))])


def fit_control(panel, matrices, mask, state, penalty):
    n = len(mask);counts = np.bincount(state * 5 + panel["records"].action.to_numpy(), minlength=n * 5).reshape(n, 5)
    initial = empirical_reward(counts, matrices, mask)
    normalizer = float(counts.sum())
    def objective(flat):
        base = flat.reshape(n, 4)
        ll, gradient, _ = policy_likelihood_gradient(_reward(base), counts, matrices, mask)
        return -ll / normalizer + penalty * np.square(base).sum() / 2, -gradient[:, :4].ravel() / normalizer + penalty * flat
    optimum = minimize(objective, initial.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": 400, "gtol": 1e-6, "ftol": 1e-10})
    base = optimum.x.reshape(n, 4)
    loss, gradient = objective(optimum.x)
    probability, value, residual = soft_policy(_reward(base), matrices, mask)
    return {"base": base, "probability": probability, "value": value,
        "fit": {"converged": bool(optimum.success), "message": str(optimum.message), "iterations": int(optimum.nit),
            "penalized_negative_log_likelihood_per_record": float(loss), "gradient_max_abs": float(np.abs(gradient).max()),
            "bellman_residual": residual}}


def component_statistics(base, delta, matrices, mask):
    statistics = [soft_policy(_reward(base, delta, sign), matrices, mask) for sign in (-1, 1)]
    probability = np.asarray([s[0] for s in statistics])
    logp = np.zeros_like(probability)
    for z in range(2):logp[z, mask] = np.log(np.maximum(probability[z, mask], 1e-300))
    return statistics, logp


def mixture_marginal(base, delta, weights, counts, matrices, mask, penalty):
    statistics, logp = component_statistics(base, delta, matrices, mask)
    user_ll = counts @ logp.reshape(2, -1).T
    joint = user_ll + np.log(np.maximum(weights, 1e-300))[None, :]
    normalizer = float(counts.sum())
    responsibility = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    raw = float(logsumexp(joint, axis=1).sum() / normalizer)
    penalized = raw - penalty * (np.square(base).sum() + np.square(delta).sum()) / 2
    return penalized, raw, responsibility, statistics


def weighted_q_objective(flat, component_counts, matrices, mask, normalizer, penalty):
    """Negative regularized EM reward Q; offsets are -delta/+delta (four free coefficients)."""
    n = len(mask);base = flat[:n * 4].reshape(n, 4);delta = flat[n * 4:]
    ll, base_gradient, delta_gradient = 0.0, np.zeros_like(base), np.zeros(4)
    for z, sign in enumerate((-1, 1)):
        value, gradient, _ = policy_likelihood_gradient(_reward(base, delta, sign), component_counts[z], matrices, mask)
        ll += value;base_gradient += gradient[:, :4];delta_gradient += sign * gradient[:, :4].sum(axis=0)
    loss = -ll / normalizer + penalty * (np.square(base).sum() + np.square(delta).sum()) / 2
    gradient = np.concatenate([-base_gradient.ravel() / normalizer + penalty * base.ravel(),
                               -delta_gradient / normalizer + penalty * delta])
    return float(loss), gradient


def fit_mixture(panel, matrices, mask, state, penalty, control):
    users, counts = customer_counts(panel, state, len(mask))
    action_counts = np.column_stack([np.asarray(counts[:, a::5].sum(axis=1)).ravel() for a in range(5)])
    composition = action_counts / action_counts.sum(axis=1, keepdims=True)
    scale = composition.std(axis=0);scale[scale < 1e-10] = 1
    labels = KMeans(n_clusters=2, n_init=5, random_state=42).fit_predict(composition / scale)
    cluster_counts = np.asarray([action_counts[labels == z].sum(axis=0) for z in range(2)])
    log_odds = np.log(np.maximum(cluster_counts[:, :4], 1e-8)) - np.log(np.maximum(cluster_counts[:, 4:5], 1e-8))
    delta = (log_odds[1] - log_odds[0]) / 2
    weights = np.bincount(labels, minlength=2).astype(float) / len(labels)
    base = control["base"].copy()
    normalizer = float(counts.sum());trace = []
    prior_penalized = None
    for iteration in range(8):
        before, raw_before, responsibility, _ = mixture_marginal(base, delta, weights, counts, matrices, mask, penalty)
        component_counts = np.asarray([(counts.T @ responsibility[:, z]).reshape(len(mask), 5) for z in range(2)])
        updated_weights = responsibility.mean(axis=0)
        prior_q_before = float((responsibility * np.log(np.maximum(weights, 1e-300))).sum() / normalizer)
        prior_q_after = float((responsibility * np.log(np.maximum(updated_weights, 1e-300))).sum() / normalizer)
        initial = np.concatenate([base.ravel(), delta])
        objective = lambda x: weighted_q_objective(x, component_counts, matrices, mask, normalizer, penalty)
        q_before = objective(initial)[0]
        optimum = minimize(objective, initial, jac=True, method="L-BFGS-B",
                           options={"maxiter": 100, "gtol": 2e-6, "ftol": 1e-10})
        q_after = objective(optimum.x)[0]
        if q_after > q_before + 1e-9:
            raise ValueError("generalized EM reward Q decreased")
        candidate_base = optimum.x[:len(mask) * 4].reshape(len(mask), 4)
        candidate_delta = optimum.x[len(mask) * 4:]
        after, raw_after, _, _ = mixture_marginal(candidate_base, candidate_delta, updated_weights, counts, matrices, mask, penalty)
        if after < before - 1e-8:
            raise ValueError("regularized mixture marginal likelihood decreased")
        base, delta, weights = candidate_base, candidate_delta, updated_weights
        trace.append({"iteration": iteration + 1, "reward_negative_Q_before": q_before, "reward_negative_Q_after": q_after,
            "prior_Q_before": prior_q_before, "prior_Q_after": prior_q_after,
            "regularized_complete_Q_before": prior_q_before - q_before,
            "regularized_complete_Q_after": prior_q_after - q_after,
            "penalized_marginal_before": before, "penalized_marginal_after": after,
            "raw_marginal_before": raw_before, "raw_marginal_after": raw_after,
            "m_step_converged": bool(optimum.success), "m_step_iterations": int(optimum.nit),
            "mixture_weights": weights.tolist()})
        if prior_penalized is not None and abs(after - prior_penalized) < 1e-6:break
        prior_penalized = after
    penalized, raw, responsibility, statistics = mixture_marginal(base, delta, weights, counts, matrices, mask, penalty)
    # The anonymous IDs have a deterministic order; this is not a personality label.
    teacher_frequency = np.bincount(state, minlength=len(mask)).astype(float)
    expected_purchase = [np.average(stat[0][:, 3], weights=teacher_frequency) for stat in statistics]
    if expected_purchase[0] > expected_purchase[1]:
        delta = -delta;weights = weights[::-1];responsibility = responsibility[:, ::-1];statistics = statistics[::-1]
    return {"base": base, "delta": delta, "weights": weights, "statistics": statistics,
        "users": users, "responsibility": responsibility, "fit": {"method": "generalized EM with exact soft-Bellman action-likelihood M-steps",
            "reward_gauge": "END reward=0; common reward table plus component offsets -delta/+delta, four free latent reward coefficients",
            "loss_denominator": "total observed checkout-choice and inferred-END records; E user likelihood is an undiscounted sum of causal action log probabilities",
            "raw_user_marginal_log_likelihood_per_record": raw, "penalized_user_marginal_log_likelihood_per_record": penalized,
            "em_iterations": len(trace), "trace": trace,
            "maximum_bellman_residual": max(s[2] for s in statistics),
            "mean_posterior_entropy": float(-np.sum(responsibility * np.log(np.maximum(responsibility, 1e-300)), axis=1).mean()),
            "effective_customer_mass": responsibility.sum(axis=0).tolist(),
            "initialization": "source pre-cutoff customer action-composition KMeans; source cluster log action/END odds initialize delta only; 1e-8 floor is numerical initialization, not a fitted probability prior"}}


def centered_offsets(free):
    return np.vstack([free, -free.sum(axis=0, keepdims=True)])


def family_statistics(base, offsets, matrices, mask):
    result = [soft_policy(np.column_stack([base + offset, np.zeros(len(base))]), matrices, mask) for offset in offsets]
    logp = np.zeros((len(offsets), len(base), 5))
    for z, item in enumerate(result):logp[z, mask] = np.log(np.maximum(item[0][mask], 1e-300))
    return result, logp


def family_marginal(base, offsets, weights, counts, matrices, mask, penalty):
    statistics, logp = family_statistics(base, offsets, matrices, mask)
    joint = counts @ logp.reshape(len(weights), -1).T + np.log(np.maximum(weights, 1e-300))[None, :]
    raw = float(logsumexp(joint, axis=1).sum() / counts.sum())
    responsibility = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    penalized = raw - penalty * np.square(base).sum() / 2 - penalty * np.square(offsets).sum() / (2 * len(weights))
    return penalized, raw, responsibility, statistics


def family_q_objective(flat, component_counts, matrices, mask, normalizer, penalty):
    n, k = len(mask), len(component_counts)
    base = flat[:n * 4].reshape(n, 4)
    offsets = centered_offsets(flat[n * 4:].reshape(k - 1, 4))
    log_likelihood, common_gradient = 0.0, np.zeros_like(base)
    offset_gradient = []
    for z in range(k):
        ll, gradient, _ = policy_likelihood_gradient(np.column_stack([base + offsets[z], np.zeros(n)]), component_counts[z], matrices, mask)
        log_likelihood += ll;common_gradient += gradient[:, :4];offset_gradient.append(gradient[:, :4].sum(axis=0))
    offset_gradient = np.asarray(offset_gradient)
    loss = -log_likelihood / normalizer + penalty * np.square(base).sum() / 2 + penalty * np.square(offsets).sum() / (2 * k)
    free_gradient = -(offset_gradient[:-1] - offset_gradient[-1]) / normalizer + penalty / k * (offsets[:-1] - offsets[-1])
    return float(loss), np.concatenate([-common_gradient.ravel() / normalizer + penalty * base.ravel(), free_gradient.ravel()])


def fit_style_family(panel, matrices, mask, state, penalty, control, components):
    users, counts = customer_counts(panel, state, len(mask))
    action_counts = np.column_stack([np.asarray(counts[:, a::5].sum(axis=1)).ravel() for a in range(5)])
    composition = action_counts / action_counts.sum(axis=1, keepdims=True)
    scale = composition.std(axis=0);scale[scale < 1e-10] = 1
    labels = KMeans(n_clusters=components, n_init=5, random_state=42).fit_predict(composition / scale)
    cluster_counts = np.asarray([action_counts[labels == z].sum(axis=0) for z in range(components)])
    odds = np.log(np.maximum(cluster_counts[:, :4], 1e-8)) - np.log(np.maximum(cluster_counts[:, 4:5], 1e-8))
    offsets = odds - odds.mean(axis=0, keepdims=True)
    weights = np.bincount(labels, minlength=components).astype(float) / len(labels)
    base = control["base"].copy();normalizer = float(counts.sum());trace = [];previous = None
    for iteration in range(8):
        before, raw_before, responsibility, _ = family_marginal(base, offsets, weights, counts, matrices, mask, penalty)
        component_counts = np.asarray([(counts.T @ responsibility[:, z]).reshape(len(mask), 5) for z in range(components)])
        updated_weights = responsibility.mean(axis=0)
        prior_before = float((responsibility * np.log(np.maximum(weights, 1e-300))).sum() / normalizer)
        prior_after = float((responsibility * np.log(np.maximum(updated_weights, 1e-300))).sum() / normalizer)
        initial = np.concatenate([base.ravel(), offsets[:-1].ravel()])
        objective = lambda x: family_q_objective(x, component_counts, matrices, mask, normalizer, penalty)
        q_before = objective(initial)[0]
        optimum = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": 100, "gtol": 2e-6, "ftol": 1e-10})
        q_after = objective(optimum.x)[0]
        if q_after > q_before + 1e-9:raise ValueError("style reward Q decreased")
        candidate_base = optimum.x[:len(mask) * 4].reshape(len(mask), 4)
        candidate_offsets = centered_offsets(optimum.x[len(mask) * 4:].reshape(components - 1, 4))
        after, raw_after, _, _ = family_marginal(candidate_base, candidate_offsets, updated_weights, counts, matrices, mask, penalty)
        if after < before - 1e-8:raise ValueError("style penalized marginal likelihood decreased")
        base, offsets, weights = candidate_base, candidate_offsets, updated_weights
        trace.append({"iteration": iteration + 1, "reward_negative_Q_before": q_before, "reward_negative_Q_after": q_after,
            "prior_Q_before": prior_before, "prior_Q_after": prior_after,
            "regularized_complete_Q_before": prior_before - q_before, "regularized_complete_Q_after": prior_after - q_after,
            "penalized_marginal_before": before, "penalized_marginal_after": after,
            "raw_marginal_before": raw_before, "raw_marginal_after": raw_after,
            "m_step_converged": bool(optimum.success), "m_step_iterations": int(optimum.nit), "mixture_weights": weights.tolist()})
        if previous is not None and abs(after - previous) < 1e-6:break
        previous = after
    penalized, raw, responsibility, statistics = family_marginal(base, offsets, weights, counts, matrices, mask, penalty)
    frequency = np.bincount(state, minlength=len(mask)).astype(float)
    order = np.argsort([np.average(s[0][:, 3], weights=frequency) for s in statistics], kind="stable")
    offsets, weights, responsibility = offsets[order], weights[order], responsibility[:, order]
    statistics = [statistics[i] for i in order]
    return {"base": base, "offsets": offsets, "weights": weights, "statistics": statistics,
        "users": users, "responsibility": responsibility, "fit": {"method": "generalized EM with exact soft-Bellman action-likelihood M-steps",
            "reward_gauge": "END reward=0; sum of component action offsets=0; 4*(K-1) free latent reward coefficients",
            "offset_regularization": "lambda/(2K)*sum_z||offset_z||^2; equals legacy lambda/2*||delta||^2 for K2 +/-delta",
            "loss_denominator": "total observed checkout-choice and inferred-END records; E user likelihood is an undiscounted sum of causal action log probabilities",
            "raw_user_marginal_log_likelihood_per_record": raw, "penalized_user_marginal_log_likelihood_per_record": penalized,
            "em_iterations": len(trace), "trace": trace, "maximum_bellman_residual": max(s[2] for s in statistics),
            "mean_posterior_entropy": float(-np.sum(responsibility * np.log(np.maximum(responsibility, 1e-300)), axis=1).mean()),
            "effective_customer_mass": responsibility.sum(axis=0).tolist(), "latent_free_reward_coefficients": 4 * (components - 1),
            "posterior_weighted_customer_action_rates": [np.average(composition, axis=0, weights=responsibility[:, z]).tolist() if responsibility[:, z].sum() > 0 else None
                                                          for z in range(components)],
            "initialization": "source pre-cutoff customer action-composition KMeans; source cluster action/END odds, centered over components; numerical 1e-8 initialization floor only"}}


def valid_record_indices(data, sessions):
    selected = data["sessions"].loc[sessions]
    return np.concatenate([np.arange(x.record_start, x.record_end) for x in selected.itertuples()])


def prequential_metrics(data, encoder, component_policies, weights, masks):
    """Posterior evidence arrives strictly after each label's target_at, including delayed END."""
    import heapq
    all_states, fallback = encode(encoder, data["features"])
    y = data["records"].action.to_numpy()
    selected = {}
    for name, mask in masks.items():
        for i in valid_record_indices(data, mask):selected[int(i)] = name
    s = data["sessions"]
    valid = ~s.invalid_total_cart & ~s.ambiguous_clock_sessions & ~s.is_censored
    indices = valid_record_indices(data, valid)
    summaries = {name: {"loss": [], "user": [], "actual": [], "predicted": [], "fallback": 0,
                        "evidence_before_first_scored_record": []} for name in masks}
    policies = np.asarray(component_policies)
    for uid, rows in data["records"].iloc[indices].groupby("user_id", sort=False):
        posterior = np.log(np.maximum(weights, 1e-300));pending = [];evidence = 0;first = set()
        for i, row in rows.iterrows():
            clock = row["at"]
            while pending and pending[0][0] < clock:
                _, _, update = heapq.heappop(pending);posterior += update;evidence += 1
            q = np.exp(posterior - logsumexp(posterior))
            probability = q @ policies[:, all_states[i], :]
            observed = probability[int(row.action)]
            if i in selected:
                name = selected[i];summary = summaries[name]
                summary["loss"].append(-np.log(max(observed, 1e-300)));summary["user"].append(uid)
                summary["actual"].append(int(row.action));summary["predicted"].append(probability)
                summary["fallback"] += int(fallback[i])
                if name not in first:summary["evidence_before_first_scored_record"].append(evidence);first.add(name)
            update = np.log(np.maximum(policies[:, all_states[i], int(row.action)], 1e-300))
            heapq.heappush(pending, (row.target_at, int(i), update))
    result = {}
    for name, summary in summaries.items():
        loss = np.asarray(summary["loss"]);actual = np.asarray(summary["actual"]);predicted = np.asarray(summary["predicted"])
        result[name] = {"records": len(loss), "users": len(set(summary["user"])),
            "prequential_action_nll": float(loss.mean()),
            "customer_macro_prequential_action_nll": float(pd.Series(loss).groupby(summary["user"]).mean().mean()),
            "observed_action_share": dict(zip(ACTIONS, (np.bincount(actual, minlength=5) / len(actual)).tolist())),
            "predicted_action_share": dict(zip(ACTIONS, predicted.mean(axis=0).tolist())),
            "encoder_fallback_records": summary["fallback"],
            "median_available_evidence_at_first_record": float(np.median(summary["evidence_before_first_scored_record"])),
            "posterior_clock": "population prior; only evidence with target_at strictly before current decision clock, including END last+30m; never static full-test-user posterior"}
    return result


def pack_matrix(matrix):
    p = matrix.tocsr()
    return {"data": p.data.tolist(), "indices": p.indices.tolist(), "indptr": p.indptr.tolist(), "shape": list(p.shape)}


def unpack_matrices(block):
    return [sparse.csr_matrix((x["data"], x["indices"], x["indptr"]), shape=tuple(x["shape"])) for x in block["transition_csr"]]


def fit_latent(rows, base, component_candidates=(2,)):
    component_candidates = list(component_candidates)
    if not component_candidates or component_candidates != sorted(set(component_candidates)) or any(k not in (2, 3, 4) for k in component_candidates):
        raise ValueError("component candidates must be unique increasing values from 2,3,4")
    penalties = PENALTIES if component_candidates == [2] else (1e-5,)
    cutoff = pd.Timestamp(base["validation"]["cutoff_at_utc"]);inner_cutoff = cutoff - pd.DateOffset(months=1)
    data = build_source(rows, base["observation_end"], int(base.get("seed", 42)))
    outer, inner, split_audit = old.split_sessions(data, cutoff, inner_cutoff, "known-total")
    inner_panel = select_panel(data, inner["train"])
    inner_encoder = fit_encoder(inner_panel)
    matrices, mask, state, dynamics_audit = fit_dynamics(inner_encoder, inner_panel)
    trials = []
    for penalty in penalties:
        control = fit_control(inner_panel, matrices, mask, state, penalty)
        for components in component_candidates:
            print(json.dumps({"stage": "inner_start", "regularization": penalty, "components": components, "states": len(mask)}), flush=True)
            mixture = (fit_mixture(inner_panel, matrices, mask, state, penalty, control) if components == 2 else
                       fit_style_family(inner_panel, matrices, mask, state, penalty, control, components))
            policies = [x[0] for x in mixture["statistics"]]
            validation = prequential_metrics(data, inner_encoder, policies, mixture["weights"], {k: v for k, v in inner.items() if k != "train"})
            score = float(np.average([v["prequential_action_nll"] for v in validation.values()], weights=[v["records"] for v in validation.values()]))
            trials.append({"regularization": penalty, "component_count": components, "pooled_inner_prequential_nll": score, "control_fit": control["fit"],
                "mixture_fit": mixture["fit"], "mixture_weights": mixture["weights"].tolist(), "validation": validation})
            print(json.dumps({"stage": "inner_done", "regularization": penalty, "components": components, "score": score, "mixture_weights": mixture["weights"].tolist()}), flush=True)
    chosen = min(trials, key=lambda x: x["pooled_inner_prequential_nll"])
    penalty = chosen["regularization"]
    panel = select_panel(data, outer["train"]);encoder = fit_encoder(panel)
    matrices, mask, state, dynamics_audit = fit_dynamics(encoder, panel)
    control = fit_control(panel, matrices, mask, state, penalty)
    components = chosen["component_count"]
    mixture = (fit_mixture(panel, matrices, mask, state, penalty, control) if components == 2 else
               fit_style_family(panel, matrices, mask, state, penalty, control, components))
    offsets = np.asarray([-mixture["delta"], mixture["delta"]]) if components == 2 else mixture["offsets"]
    policies = np.asarray([s[0] for s in mixture["statistics"]])
    validation = prequential_metrics(data, encoder, policies, mixture["weights"], outer)
    control_validation = prequential_metrics(data, encoder, control["probability"][None, :, :], np.ones(1), outer)
    posteriors = {hashlib.sha256(f"cosmetics:{uid}".encode()).hexdigest()[:16]: q.tolist()
                  for uid, q in zip(mixture["users"], mixture["responsibility"])}
    return {"schema_version": 1, "type": TYPE, "actions": list(ACTIONS), "discount": DISCOUNT,
        "regularization": penalty, "state_count": len(mask), "feature_names": FEATURE_NAMES,
        "context_keys": CONTEXT_KEYS, "encoder": encoder, "action_mask": mask.tolist(),
        "transition_csr": [pack_matrix(p) for p in matrices], "mixture_weights": mixture["weights"].tolist(),
        "component_count": components, "common_reward": mixture["base"].tolist(), "component_offsets": offsets.tolist(),
        "component_policies": policies.tolist(), "component_values": [s[1].tolist() for s in mixture["statistics"]],
        "profile_posteriors": posteriors, "posterior_bank_scope": "training users; source labels available strictly before outer cutoff; hashed donor profile ID; no heldout future histories or home-group labels",
        "fit": mixture["fit"], "source_audit": data["audit"], "dynamics_audit": dynamics_audit,
        "split_audit": split_audit, "training": {"seed": int(base.get("seed", 42)), "cutoff_at": cutoff.isoformat(),
            "inner_cutoff_at": inner_cutoff.isoformat(), "regularization_candidates": trials,
            "component_candidates": component_candidates, "selected_component_count": components,
            "regularization_grid": list(penalties),
            "regularization_recipe": "legacy K2 inner lambda selection" if component_candidates == [2] else "style-component search fixes source lambda=1e-5; no rollout tuning",
            "training_records": len(panel["records"]), "training_sessions": panel["sessions"], "training_users": panel["users"],
            "selection": "record-pooled strict prequential causal action NLL on three inner user/time panels; outer panels not used"},
        "validation": validation, "single_reward_control": {"fit": control["fit"], "validation": control_validation,
            "reward": control["base"].tolist(), "policy": control["probability"].tolist()},
        "choice_unit": "one checkout decision; contiguous same-clock purchased units collapsed; END is internal, not a raw event",
        "history_clock": "completed session last_action+30m strictly before decision clock; source and runtime use ShoppingHistory",
        "method_references": ["https://icml.cc/Conferences/2011/papers/478_icmlpaper.pdf",
            "https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf"],
        "runtime_latent": "draw once conditional on train-prefix donor posterior, else fitted population prior; persistent customer state, never a public 45-feature input",
        "limitations": ["Anonymous mixture components are fitted behavioral heterogeneity, not validated psychological personas.",
            "Shared source transition kernel is a partially observed finite abstraction; unmatched-SKU remove actions stay in likelihood but their transition pairs are excluded. Runtime full-inventory dynamics may differ.",
            "Donor posteriors use training-prefix traces only. Unseen donor profiles use the fitted prior; no future home-group or holdout outcome is a gate.",
            "Likelihood is measured per checkout decision, while purchased units and basket/price transport require separate generated-sequence checks.",
            "Between-session return timing and source basket banks are not jointly refitted here; donor-conditional latent draws preserve available source association but do not identify its causes.",
            "No observed cross-source person links jointly identify search/review/CS effects on this shopping reward; no external real-customer validation has been performed."]}


def _block(model):
    return model.get("latent_shopping", model)


def trait_invariant_sha256(block):
    """Fingerprint every parent block field outside donor-assignment bookkeeping."""
    ignored = {"profile_posteriors", "posterior_bank_scope", "profile_posterior_assignment_scope",
               "runtime_latent", "trait_reconstruction"}
    frozen = {key: value for key, value in block.items() if key not in ignored}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _validate_trait_bank(block):
    scope = block.get("profile_posterior_assignment_scope")
    provenance = block.get("trait_reconstruction")
    if scope is None and provenance is None:
        return
    if scope != "source_observation_trait_posterior" or not isinstance(provenance, dict):
        raise ValueError("invalid synthesis-only trait assignment scope")
    if (provenance.get("schema_version") != 1 or provenance.get("purpose") != "synthesis_only"
            or provenance.get("forecast_validation_uses_bank") is not False):
        raise ValueError("source trait bank must remain outside forecast validation")
    for key in ("parent_model_sha256", "source_cache_sha256", "immutable_parent_block_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("missing source trait reconstruction fingerprint")
    if provenance["immutable_parent_block_sha256"] != trait_invariant_sha256(block):
        raise ValueError("source trait reconstruction changed fitted parent parameters or validation")
    count = provenance.get("posterior_profiles")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(block["profile_posteriors"]) or count < 1:
        raise ValueError("source trait posterior profile count differs")
    for key in ("observation_start_at", "observation_end_at"):
        clock = pd.Timestamp(provenance.get(key))
        if pd.isna(clock) or clock.tzinfo is None:
            raise ValueError("source trait observation clocks must be explicit UTC")
    if pd.Timestamp(provenance["observation_start_at"]) >= pd.Timestamp(provenance["observation_end_at"]):
        raise ValueError("source trait observation bounds differ")


def validate_model(model):
    b = _block(model)
    if b.get("schema_version") != 1 or b.get("type") != TYPE or b.get("actions") != list(ACTIONS) or b.get("feature_names") != FEATURE_NAMES:
        raise ValueError("incompatible latent shopping model")
    n = b["state_count"]
    if not isinstance(n, int) or not 0 < n <= 512 or len(b["encoder"]["state_keys"]) != n or b["discount"] != DISCOUNT:
        raise ValueError("invalid latent state count/discount")
    if not np.isfinite(b["regularization"]) or b["regularization"] <= 0:
        raise ValueError("invalid latent reward regularization")
    encoder = b["encoder"];keys = np.asarray(encoder["state_keys"])
    means = np.asarray(encoder["context_means"]);scale = np.asarray(encoder["feature_scale"])
    if encoder.get("feature_names") != FEATURE_NAMES or keys.shape != (n, 2) or len({tuple(k) for k in keys}) != n or (keys[:, 1] < 0).any() or (keys[:, 1] > 7).any():
        raise ValueError("invalid observed state encoder")
    if means.shape != (n, len(FEATURE_NAMES)) or scale.shape != (len(FEATURE_NAMES),) or not np.isfinite(means).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid observed context statistics")
    tree = encoder["tree"];node_count = len(tree["left"])
    if not node_count or not all(len(tree[k]) == node_count for k in ("right", "feature", "threshold")):
        raise ValueError("invalid observed tree arrays")
    for i, (left, right, feature, threshold) in enumerate(zip(tree["left"], tree["right"], tree["feature"], tree["threshold"])):
        if left >= 0 and not (i < left < node_count and i < right < node_count and 0 <= feature < len(FEATURE_NAMES) and np.isfinite(threshold)):
            raise ValueError("invalid observed tree branch")
    matrices = unpack_matrices(b);mask = np.asarray(b["action_mask"], dtype=bool)
    if len(matrices) != 5 or mask.shape != (n, 5) or not np.array_equal(mask, action_mask(b["encoder"])):
        raise ValueError("invalid latent action support")
    for a, p in enumerate(matrices):
        expected = mask[:, a].astype(float) if a != 4 else np.zeros(n)
        if p.shape != (n, n) or not np.isfinite(p.data).all() or (p.data < 0).any() or not np.allclose(np.asarray(p.sum(axis=1)).ravel(), expected):
            raise ValueError("invalid source transition matrix")
    weights = np.asarray(b["mixture_weights"]);base = np.asarray(b["common_reward"]);offsets = np.asarray(b["component_offsets"])
    components = len(weights)
    policies = np.asarray(b["component_policies"])
    if components not in (2, 3, 4) or base.shape != (n, 4) or offsets.shape != (components, 4) or policies.shape != (components, n, 5) or b.get("component_count", components) != components:
        raise ValueError("invalid mixture array shape")
    if not all(np.isfinite(x).all() for x in [weights, base, offsets, policies]) or (weights < 0).any() or not np.isclose(weights.sum(), 1) or not np.allclose(offsets.sum(axis=0), 0) or (policies < 0).any() or not np.allclose(policies.sum(axis=2), 1) or policies[:, ~mask].any():
        raise ValueError("invalid learned mixture weights/gauge")
    for z in range(components):
        probability, value, residual = soft_policy(np.column_stack([base + offsets[z], np.zeros(n)]), matrices, mask)
        if not np.allclose(probability, policies[z], atol=1e-8, rtol=1e-8) or not np.allclose(value, b["component_values"][z], atol=1e-8, rtol=1e-8):
            raise ValueError("saved latent policy does not reproduce its reward and dynamics")
    for q in b["profile_posteriors"].values():
        if len(q) != components or not np.isfinite(q).all() or min(q) < 0 or not np.isclose(sum(q), 1):
            raise ValueError("invalid source-prefix posterior")
    _validate_trait_bank(b)


def policy(model, context, component_id):
    b = _block(model)
    if isinstance(component_id, bool) or not isinstance(component_id, (int, np.integer)) or not 0 <= component_id < len(b["mixture_weights"]):
        raise ValueError("component_id must be the persistent private customer draw")
    cache = _RUNTIME.get(id(b))
    if cache is None or cache[0] is not b:
        encoder = b["encoder"];tree = encoder["tree"];keys = np.asarray(encoder["state_keys"])
        cache = (b, {tuple(k): i for i, k in enumerate(keys)}, keys,
            np.asarray(encoder["context_means"]), np.asarray(encoder["feature_scale"]),
            np.asarray(b["component_policies"]),
            tuple(np.asarray(tree[name]) for name in ("left", "right", "feature", "threshold")))
        if len(_RUNTIME) >= 4:_RUNTIME.clear()
        _RUNTIME[id(b)] = cache
    _, mapping, keys, means, scale, probabilities, tree = cache
    x = context_vector(context);xf = x.astype(np.float32);left, right, feature, threshold = tree
    node = 0
    while left[node] >= 0:node = left[node] if xf[feature[node]] <= threshold[node] else right[node]
    group = cart_bucket(context.get("cart_total_quantity", 0)) * 2 + int(context.get("session_decision_count", 0) == 0)
    state = mapping.get((int(node), group))
    if state is None:
        candidates = np.flatnonzero(keys[:, 1] == group)
        if not len(candidates):raise ValueError("no source state supports runtime cart/START group")
        state = int(candidates[np.argmin(np.square((means[candidates] - x) / scale).sum(axis=1))])
    values = probabilities[int(component_id), state].copy()
    if context.get("session_decision_count", 0) == 0:values[4] = 0
    if context.get("cart_total_quantity", 0) == 0:values[2] = 0
    values /= values.sum()
    return dict(zip(ACTIONS, values.tolist()))


def sample_component(model, profile_id, rng):
    b = _block(model)
    probability = b["profile_posteriors"].get(profile_id)
    scope = b.get("profile_posterior_assignment_scope", "train_prefix_profile_posterior")
    if probability is None:probability, scope = b["mixture_weights"], "population_prior"
    return int(rng.choice(len(b["mixture_weights"]), p=probability)), scope


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/models/shopping_profiles_aligned_v6.json"))
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/models/shopping_latent_irl_candidate.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/audit/simulator/latent_shopping_training.json"))
    parser.add_argument("--component-candidates", type=int, nargs="+", choices=(2, 3, 4), default=[2])
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():raise ValueError("model and report paths must differ")
    for p in (args.output, args.report):
        if p.exists():raise FileExistsError(f"immutable fit output exists: {p}")
    base = json.loads(args.base.read_text())
    if "causal_shopping" in base or "latent_shopping" in base:raise ValueError("latent IRL needs the original observed shopping base, not another actor")
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    rows, source = load_prior_source(base, args.source_cache)
    block = fit_latent(rows, base, args.component_candidates)
    block["provenance"] = {"base_path": str(args.base), "base_sha256": digest(args.base), "source_cache": source,
        "source_cache_sha256": digest(args.source_cache) if args.source_cache.is_file() else None,
        "fitter_sha256": digest(__file__), "history_helper_sha256": digest(Path(__file__).with_name("shopping_history.py"))}
    validate_model(block)
    artifact = deepcopy(base);artifact["latent_shopping"] = block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:json.dump(artifact, stream, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    report = {"passed": True, "model_path": str(args.output), "model_sha256": digest(args.output),
        "type": TYPE, "provenance": block["provenance"], "training": block["training"], "fit": block["fit"],
        "state_count": block["state_count"], "mixture_weights": block["mixture_weights"],
        "component_offsets": block["component_offsets"], "source_audit": block["source_audit"],
        "dynamics_audit": block["dynamics_audit"], "split_audit": block["split_audit"],
        "validation": block["validation"], "single_reward_control": block["single_reward_control"],
        "limitations": block["limitations"], "adoption": "candidate; generated-sequence quality and raw integrity must be evaluated separately"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"model_path": str(args.output), "model_sha256": report["model_sha256"],
                      "state_count": block["state_count"], "mixture_weights": block["mixture_weights"]}), flush=True)


if __name__ == "__main__":main()
