"""Customer-shared shopping IRL and observed 30-day activity likelihood.

The shopping policy solves the existing soft Bellman model.  The activity head
is a supervised, fixed-base logit with source-fitted component intercepts; it is
not an identified long-horizon return reward.  No old actor is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.cluster import KMeans

from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl.fit_return import return_context, return_probability


policy = shopping.policy
sample_component = shopping.sample_component
TYPE = shopping.TYPE
COMPONENTS = 4
PENALTY = 1e-5
CUTOFF = "2020-01-29T00:00:00+00:00"
ACTIVITY_TYPE = "shared_component_fixed_base_logit_offset"


def _block(model):
    return model.get("latent_shopping", model)


def baseline_logit(model, context):
    """Match engine source_return_logit, including its probability bounds."""
    probability = np.clip(return_probability(model, context), 1e-5, 1 - 1e-5)
    return float(np.log(probability / (1 - probability)))


def build_activity_panel(rows, return_model, seed=42):
    """Non-overlapping calendar windows after a customer's first observation.

    A first-observed session is never an activity label in its own entry window.
    Inactive windows remain observations. All features are strictly pre-as-of.
    """
    start = rows["at"].min().floor("D")
    end = rows["at"].max().floor("D") + pd.Timedelta(days=1)
    snapshots = list(pd.date_range(start + pd.Timedelta(days=30),
                                   end - pd.Timedelta(days=30), freq="30D"))
    result = []
    for uid, user in rows.groupby("user_id", sort=True):
        starts = user.groupby("session", sort=False)["at"].min().sort_values().tolist()
        events = [{"event_at": row.at, "event_type": row.event_type,
                   "session_id": str(row.session),
                   "unit_price_minor": None if pd.isna(row.price_minor) else int(row.price_minor)}
                  for row in user.itertuples(index=False)]
        train_user = hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205
        for at in snapshots:
            if starts[0] >= at:
                continue
            context = return_context(starts, events, at)
            result.append({"user_id": uid, "at": at, "target_at": at + pd.Timedelta(days=30),
                "label": int(any(at <= value < at + pd.Timedelta(days=30) for value in starts)),
                "base_logit": baseline_logit(return_model, context), "train_user": train_user,
                "active30d": bool(context["session_count_30d"] > 0)})
    if not result:
        raise ValueError("no fully observed non-entry activity windows")
    return pd.DataFrame(result), {"horizon_days": 30,
        "observation_start": start.isoformat(), "observation_end": end.isoformat(),
        "snapshots": [at.isoformat() for at in snapshots],
        "window_rule": "nonoverlapping [as_of,as_of+30d); first_observed_session < as_of; full followup required",
        "context_rule": "existing return_context strictly past source events and starts; exact runtime clipped baseline logit"}


def activity_masks(activity, cutoff):
    early = activity.target_at.lt(cutoff)
    late = activity["at"].ge(cutoff)
    return {"train": early & activity.train_user,
            "user_holdout": early & ~activity.train_user,
            "time_holdout": late & activity.train_user,
            "user_and_time_holdout": late & ~activity.train_user}


def activity_log_probability(activity, offsets):
    eta = activity.base_logit.to_numpy()[:, None] + np.asarray(offsets)[None, :]
    return activity.label.to_numpy()[:, None] * eta - np.logaddexp(0, eta)


def joint_counts(panel, state, states, activity):
    action_users, action_counts = shopping.customer_counts(panel, state, states)
    users = np.union1d(action_users, activity.user_id.unique())
    action_coo = action_counts.tocoo()
    counts = sparse.coo_matrix((action_coo.data,
        (np.searchsorted(users, action_users)[action_coo.row], action_coo.col)),
        shape=(len(users), states * 5)).tocsr()
    activity_user = np.searchsorted(users, activity.user_id.to_numpy())
    incidence = sparse.coo_matrix((np.ones(len(activity)),
        (activity_user, np.arange(len(activity)))), shape=(len(users), len(activity))).tocsr()
    return users, counts, incidence, activity_user


def joint_marginal(base, offsets, return_offsets, weights, counts, incidence,
                   activity, matrices, mask, penalty):
    statistics, log_policy = shopping.family_statistics(base, offsets, matrices, mask)
    user_ll = counts @ log_policy.reshape(len(weights), -1).T
    user_ll += incidence @ activity_log_probability(activity, return_offsets)
    joint = user_ll + np.log(np.maximum(weights, 1e-300))[None, :]
    normalizer = float(counts.sum() + len(activity))
    raw = float(logsumexp(joint, axis=1).sum() / normalizer)
    responsibility = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
    regularizer = penalty * np.square(base).sum() / 2
    regularizer += penalty * (np.square(offsets).sum() + np.square(return_offsets).sum()) / (2 * len(weights))
    return raw - regularizer, raw, responsibility, statistics


def joint_q_objective(flat, component_counts, activity, activity_responsibility,
                      matrices, mask, normalizer, penalty):
    n, components = len(mask), len(component_counts)
    reward_length = n * 4 + (components - 1) * 4
    loss, reward_gradient = shopping.family_q_objective(flat[:reward_length],
        component_counts, matrices, mask, normalizer, penalty)
    offsets = flat[reward_length:]
    log_probability = activity_log_probability(activity, offsets)
    loss -= float((activity_responsibility * log_probability).sum() / normalizer)
    loss += penalty * np.square(offsets).sum() / (2 * components)
    probability = expit(activity.base_logit.to_numpy()[:, None] + offsets[None, :])
    gradient = (activity_responsibility * (probability - activity.label.to_numpy()[:, None])).sum(axis=0) / normalizer
    gradient += penalty / components * offsets
    return float(loss), np.concatenate([reward_gradient, gradient])


def unpack_parameters(flat, states, components):
    base = flat[:states * 4].reshape(states, 4)
    end = states * 4 + (components - 1) * 4
    offsets = shopping.centered_offsets(flat[states * 4:end].reshape(components - 1, 4))
    return base, offsets, flat[end:]


def fit_control(panel, matrices, mask, state, activity):
    _, counts, _, _ = joint_counts(panel, state, len(mask), activity)
    total = np.asarray(counts.sum(axis=0)).reshape(len(mask), 5)
    initial = np.concatenate([shopping.empirical_reward(total, matrices, mask).ravel(), [0.]])
    normalizer = float(total.sum() + len(activity))
    objective = lambda flat: joint_q_objective(flat, total[None], activity,
        np.ones((len(activity), 1)), matrices, mask, normalizer, PENALTY)
    optimum = minimize(objective, initial, jac=True, method="L-BFGS-B",
        options={"maxiter": 400, "gtol": 1e-6, "ftol": 1e-10})
    base, offsets, return_offsets = unpack_parameters(optimum.x, len(mask), 1)
    probability, value, residual = shopping.soft_policy(shopping._reward(base), matrices, mask)
    loss, gradient = objective(optimum.x)
    return {"base": base, "return_offsets": return_offsets,
        "statistics": [(probability, value, residual)], "fit": {
            "converged": bool(optimum.success), "message": str(optimum.message),
            "iterations": int(optimum.nit), "penalized_negative_log_likelihood_per_modeled_observation": loss,
            "gradient_max_abs": float(np.abs(gradient).max()), "bellman_residual": residual}}


def fit_joint_mixture(panel, matrices, mask, state, activity, control, max_em=8, max_m=100):
    users, counts, incidence, activity_user = joint_counts(panel, state, len(mask), activity)
    action_counts = np.column_stack([np.asarray(counts[:, a::5].sum(axis=1)).ravel() for a in range(5)])
    action_exposure = action_counts.sum(axis=1)
    composition = np.divide(action_counts, action_exposure[:, None],
        out=np.zeros_like(action_counts), where=action_exposure[:, None] > 0)
    exposure = np.asarray(incidence.sum(axis=1)).ravel()
    positive = np.asarray(incidence @ activity.label.to_numpy()).ravel()
    rates = np.divide(positive, exposure, out=np.zeros_like(positive), where=exposure > 0)
    initialization = np.column_stack([composition, rates, np.log1p(exposure), exposure == 0, action_exposure == 0])
    scale = initialization.std(axis=0);scale[scale < 1e-10] = 1
    labels = KMeans(n_clusters=COMPONENTS, n_init=5, random_state=42).fit_predict(initialization / scale)
    cluster_counts = np.asarray([action_counts[labels == z].sum(axis=0) for z in range(COMPONENTS)])
    odds = np.log(np.maximum(cluster_counts[:, :4], 1e-8)) - np.log(np.maximum(cluster_counts[:, 4:5], 1e-8))
    offsets = odds - odds.mean(axis=0, keepdims=True)
    weights = np.bincount(labels, minlength=COMPONENTS).astype(float) / len(labels)
    return_offsets = np.zeros(COMPONENTS)
    base = control["base"].copy();normalizer = float(counts.sum() + len(activity))
    # Initialize the return coefficients by the same source likelihood, not a probability prior.
    hard = np.eye(COMPONENTS)[labels[activity_user]]
    def return_initial_objective(value):
        probability = expit(activity.base_logit.to_numpy()[:, None] + value[None, :])
        loss = -(hard * activity_log_probability(activity, value)).sum() / normalizer + PENALTY * np.square(value).sum() / (2 * COMPONENTS)
        gradient = (hard * (probability - activity.label.to_numpy()[:, None])).sum(axis=0) / normalizer + PENALTY / COMPONENTS * value
        return float(loss), gradient
    return_offsets = minimize(return_initial_objective, return_offsets, jac=True, method="L-BFGS-B",
        options={"maxiter": 100, "gtol": 1e-8, "ftol": 1e-12}).x
    trace = [];previous = None
    for iteration in range(max_em):
        before, raw_before, responsibility, _ = joint_marginal(base, offsets, return_offsets,
            weights, counts, incidence, activity, matrices, mask, PENALTY)
        component_counts = np.asarray([(counts.T @ responsibility[:, z]).reshape(len(mask), 5) for z in range(COMPONENTS)])
        activity_responsibility = responsibility[activity_user]
        updated_weights = responsibility.mean(axis=0)
        prior_before = float((responsibility * np.log(np.maximum(weights, 1e-300))).sum() / normalizer)
        prior_after = float((responsibility * np.log(np.maximum(updated_weights, 1e-300))).sum() / normalizer)
        initial = np.concatenate([base.ravel(), offsets[:-1].ravel(), return_offsets])
        objective = lambda value: joint_q_objective(value, component_counts, activity,
            activity_responsibility, matrices, mask, normalizer, PENALTY)
        q_before = objective(initial)[0]
        optimum = minimize(objective, initial, jac=True, method="L-BFGS-B",
            options={"maxiter": max_m, "gtol": 2e-6, "ftol": 1e-10})
        q_after, gradient = objective(optimum.x)
        if q_after > q_before + 1e-9:raise ValueError("joint regularized Q decreased")
        candidate = unpack_parameters(optimum.x, len(mask), COMPONENTS)
        after, raw_after, _, _ = joint_marginal(*candidate, updated_weights, counts,
            incidence, activity, matrices, mask, PENALTY)
        if after < before - 1e-8:raise ValueError("joint penalized marginal likelihood decreased")
        base, offsets, return_offsets = candidate;weights = updated_weights
        trace.append({"iteration": iteration + 1, "negative_Q_before": q_before, "negative_Q_after": q_after,
            "prior_Q_before": prior_before, "prior_Q_after": prior_after,
            "regularized_complete_Q_before": prior_before - q_before,
            "regularized_complete_Q_after": prior_after - q_after,
            "penalized_marginal_before": before, "penalized_marginal_after": after,
            "raw_marginal_before": raw_before, "raw_marginal_after": raw_after,
            "m_step_converged": bool(optimum.success), "m_step_message": str(optimum.message),
            "m_step_iterations": int(optimum.nit), "m_step_gradient_max_abs": float(np.abs(gradient).max()),
            "mixture_weights": weights.tolist(), "return_logit_offsets": return_offsets.tolist()})
        print(json.dumps({"stage": "joint_em", **trace[-1]}), flush=True)
        if previous is not None and abs(after - previous) < 1e-6:break
        previous = after
    penalized, raw, responsibility, statistics = joint_marginal(base, offsets, return_offsets,
        weights, counts, incidence, activity, matrices, mask, PENALTY)
    frequency = np.bincount(state, minlength=len(mask)).astype(float)
    order = np.argsort([np.average(s[0][:, 3], weights=frequency) for s in statistics], kind="stable")
    offsets, return_offsets, weights = offsets[order], return_offsets[order], weights[order]
    responsibility = responsibility[:, order];statistics = [statistics[z] for z in order]
    return {"base": base, "offsets": offsets, "return_offsets": return_offsets, "weights": weights,
        "statistics": statistics, "users": users, "responsibility": responsibility,
        "fit": {"method": "generalized EM: shared customer likelihood of soft-Bellman actions and fixed-base activity logit",
            "objective": "sum_i log(sum_z w_z exp(sum_t log pi_z(a_it|s_it) + sum_j [y_ij*(base_logit_ij+b_z)-softplus(base_logit_ij+b_z)]))/(N_action+N_activity) - regularizer",
            "reward_gauge": "END reward=0; component shopping action offsets sum to zero",
            "return_gauge": "K free activity offsets; frozen source base intercept; no centering constraint",
            "regularizer": "lambda/2*||common_reward||^2 + lambda/(2K)*(sum||shopping_offset||^2+sum activity_offset^2)",
            "normalizer": normalizer, "action_observations": int(counts.sum()), "activity_observations": len(activity),
            "raw_user_marginal_log_likelihood_per_modeled_observation": raw,
            "penalized_user_marginal_log_likelihood_per_modeled_observation": penalized,
            "em_iterations": len(trace), "em_iteration_budget": max_em, "m_step_iteration_budget": max_m,
            "termination": "GEM budget or marginal-change criterion; not a claim of a converged global optimum",
            "trace": trace, "maximum_bellman_residual": max(s[2] for s in statistics),
            "effective_customer_mass": responsibility.sum(axis=0).tolist(),
            "mean_posterior_entropy": float(-np.sum(responsibility * np.log(np.maximum(responsibility, 1e-300)), axis=1).mean()),
            "latent_free_reward_coefficients": 4 * (COMPONENTS - 1), "activity_coefficients": COMPONENTS,
            "initialization": "source action compositions, matured activity rate, log exposure and missing-exposure flags; KMeans seed42; source weighted-logit activity initialization; 1e-8 shopping floor only for initialization",
            "posterior_weighted_customer_action_rates": [np.average(composition, axis=0, weights=responsibility[:, z]).tolist() for z in range(COMPONENTS)],
            "component_order": "anonymous IDs sorted by train teacher purchase probability; no psychological/persona interpretation"}}


def prequential_metrics(data, encoder, policies, weights, activity, return_offsets,
                       shopping_masks, activity_split_masks):
    """Both outcomes become evidence only strictly after their availability clock."""
    states, fallback = shopping.encode(encoder, data["features"])
    sessions = data["sessions"]
    valid = ~sessions.invalid_total_cart & ~sessions.ambiguous_clock_sessions & ~sessions.is_censored
    indices = shopping.valid_record_indices(data, valid)
    action_selected = {int(i): name for name, mask in shopping_masks.items()
                       for i in shopping.valid_record_indices(data, mask)}
    activity_selected = {int(i): name for name, mask in activity_split_masks.items() for i in activity.index[mask]}
    names = list(shopping_masks)
    summaries = {name: {kind: {"loss": [], "baseline_loss": [], "users": [], "actual": [],
        "predicted": [], "fallback": 0, "evidence": []} for kind in ("action", "activity")} for name in names}
    action_groups = {uid: group for uid, group in data["records"].iloc[indices].groupby("user_id", sort=False)}
    activity_groups = {uid: group for uid, group in activity.groupby("user_id", sort=False)}
    policies = np.asarray(policies);weights = np.asarray(weights)
    for uid in np.union1d(list(action_groups), list(activity_groups)):
        queries = []
        if uid in action_groups:
            queries.extend((row["at"], 0, int(i), row) for i, row in action_groups[uid].iterrows())
        if uid in activity_groups:
            queries.extend((row["at"], 1, int(i), row) for i, row in activity_groups[uid].iterrows())
        queries.sort(key=lambda item: item[:3])
        posterior = np.log(np.maximum(weights, 1e-300));pending = [];serial = 0;first = set();evidence = [0, 0]
        for clock, kind, i, row in queries:
            while pending and pending[0][0] < clock:
                _, _, source_kind, update = heapq.heappop(pending)
                posterior += update;evidence[source_kind] += 1
            q = np.exp(posterior - logsumexp(posterior))
            if kind == 0:
                probability = q @ policies[:, states[i]]
                update = np.log(np.maximum(policies[:, states[i], int(row.action)], 1e-300))
                selected = action_selected.get(i);category = "action"
                actual = int(row.action);loss = -np.log(max(probability[actual], 1e-300));base_loss = None
            else:
                component_probability = expit(float(row.base_logit) + return_offsets)
                probability = float(q @ component_probability);actual = int(row.label)
                eta = float(row.base_logit) + return_offsets
                update = actual * eta - np.logaddexp(0, eta)
                selected = activity_selected.get(i);category = "activity"
                loss = -np.log(max(probability if actual else 1 - probability, 1e-300))
                base_loss = float(np.logaddexp(0, row.base_logit) - actual * row.base_logit)
            if selected is not None:
                summary = summaries[selected][category]
                summary["loss"].append(loss);summary["users"].append(uid);summary["actual"].append(actual)
                summary["predicted"].append(probability)
                if base_loss is not None:summary["baseline_loss"].append(base_loss)
                else:summary["fallback"] += int(fallback[i])
                key = (selected, category)
                if key not in first:summary["evidence"].append(evidence.copy());first.add(key)
            heapq.heappush(pending, (row.target_at, serial, kind, update));serial += 1
    result = {}
    for name, channels in summaries.items():
        result[name] = {}
        for kind, summary in channels.items():
            actual = np.asarray(summary["actual"]);prediction = np.asarray(summary["predicted"])
            if not len(actual):
                result[name][kind] = {"records": 0, "users": 0, "prequential_nll": None};continue
            value = {"records": len(actual), "users": len(set(summary["users"])),
                "prequential_nll": float(np.mean(summary["loss"])),
                "customer_macro_prequential_nll": float(pd.Series(summary["loss"]).groupby(summary["users"]).mean().mean()),
                "median_available_action_activity_evidence_before_first_record": np.median(summary["evidence"], axis=0).tolist()}
            if kind == "action":
                value.update(observed_action_share=dict(zip(shopping.ACTIONS, (np.bincount(actual, minlength=5) / len(actual)).tolist())),
                    predicted_action_share=dict(zip(shopping.ACTIONS, prediction.mean(axis=0).tolist())), encoder_fallback_records=summary["fallback"])
            else:
                value.update(observed_activity_rate=float(actual.mean()), mean_probability=float(prediction.mean()),
                    brier=float(np.square(prediction - actual).mean()), fixed_base_nll=float(np.mean(summary["baseline_loss"])))
            result[name][kind] = value
        result[name]["posterior_clock"] = "population prior; action target (including END last+30m) and activity window_end strictly before each prediction; no static heldout posterior"
    return result


def validate_model(model, return_model_sha256=None):
    shopping.validate_model(model)
    block = _block(model);head = block.get("return_activity")
    if head is None:
        return
    if not isinstance(head, dict):
        raise ValueError("invalid latent activity head")
    values = np.asarray(head.get("component_logit_offsets"), dtype=float)
    sha = head.get("base_model_sha256")
    if head.get("schema_version") != 1 or head.get("type") != ACTIVITY_TYPE or head.get("horizon_days") != 30:
        raise ValueError("incompatible latent activity head")
    if values.shape != (len(block["mixture_weights"]),) or not np.isfinite(values).all():
        raise ValueError("invalid component activity offsets")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("missing frozen return model SHA256")
    if return_model_sha256 is not None and sha != return_model_sha256:
        raise ValueError("latent activity head requires its exact frozen return model")
    if head.get("cutoff_at") != pd.Timestamp(CUTOFF).isoformat():
        raise ValueError("incompatible common shopping/activity cutoff")
    if head.get("fit_label_available_rule") != "window_end < cutoff" or head.get("posterior_available_rule") != "window_end < decision_clock":
        raise ValueError("incompatible activity availability clock")


def return_logit_offset(model, component_id):
    block = _block(model)
    if isinstance(component_id, bool) or not isinstance(component_id, (int, np.integer)) or not 0 <= component_id < len(block["mixture_weights"]):
        raise ValueError("component_id must be the persistent private customer draw")
    head = block.get("return_activity")
    return 0. if head is None else float(head["component_logit_offsets"][int(component_id)])


def fit_joint(rows, base, return_model, return_model_sha256):
    cutoff = pd.Timestamp(CUTOFF)
    if pd.Timestamp(return_model["validation"]["temporal_cutoff"]) != cutoff:
        raise ValueError("joint recipe requires the frozen Jan29 return model")
    data = shopping.build_source(rows, base["observation_end"], int(base.get("seed", 42)))
    outer, _, split_audit = shopping.old.split_sessions(data, cutoff, cutoff - pd.DateOffset(months=1), "known-total")
    activity, activity_audit = build_activity_panel(rows, return_model, int(base.get("seed", 42)))
    activity_split = activity_masks(activity, cutoff)
    training_activity = activity.loc[activity_split["train"]].reset_index(drop=True)
    if training_activity.label.nunique() != 2:raise ValueError("joint activity fit needs both outcomes")
    panel = shopping.select_panel(data, outer["train"]);encoder = shopping.fit_encoder(panel)
    matrices, mask, state, dynamics_audit = shopping.fit_dynamics(encoder, panel)
    print(json.dumps({"stage": "joint_source", "action_records": len(panel["records"]),
        "activity_records": len(training_activity), "states": len(mask)}), flush=True)
    control = fit_control(panel, matrices, mask, state, training_activity)
    mixture = fit_joint_mixture(panel, matrices, mask, state, training_activity, control)
    policies = np.asarray([s[0] for s in mixture["statistics"]])
    validation = prequential_metrics(data, encoder, policies, mixture["weights"], activity,
        mixture["return_offsets"], outer, activity_split)
    control_validation = prequential_metrics(data, encoder, np.asarray([control["statistics"][0][0]]),
        np.ones(1), activity, control["return_offsets"], outer, activity_split)
    posteriors = {hashlib.sha256(f"cosmetics:{uid}".encode()).hexdigest()[:16]: q.tolist()
                  for uid, q in zip(mixture["users"], mixture["responsibility"])}
    activity_audit["split_counts"] = {name: {"records": int(selected.sum()),
        "users": int(activity.loc[selected, "user_id"].nunique()), "positive": int(activity.loc[selected, "label"].sum()),
        "active30d": int(activity.loc[selected, "active30d"].sum())} for name, selected in activity_split.items()}
    covered = np.logical_or.reduce([m.to_numpy() for m in activity_split.values()])
    activity_audit["purged_records"] = int((~covered).sum())
    return {"schema_version": 1, "type": TYPE, "actions": list(shopping.ACTIONS),
        "discount": shopping.DISCOUNT, "regularization": PENALTY, "state_count": len(mask),
        "feature_names": shopping.FEATURE_NAMES, "context_keys": shopping.CONTEXT_KEYS,
        "encoder": encoder, "action_mask": mask.tolist(), "transition_csr": [shopping.pack_matrix(p) for p in matrices],
        "mixture_weights": mixture["weights"].tolist(), "component_count": COMPONENTS,
        "common_reward": mixture["base"].tolist(), "component_offsets": mixture["offsets"].tolist(),
        "component_policies": policies.tolist(), "component_values": [s[1].tolist() for s in mixture["statistics"]],
        "profile_posteriors": posteriors,
        "posterior_bank_scope": "training users only; shopping terminal and activity window_end strictly before Jan29; persistent anonymous customer effect",
        "return_activity": {"schema_version": 1, "type": ACTIVITY_TYPE, "horizon_days": 30,
            "base_model_sha256": return_model_sha256, "component_logit_offsets": mixture["return_offsets"].tolist(),
            "cutoff_at": cutoff.isoformat(), "fit_label_available_rule": "window_end < cutoff",
            "posterior_available_rule": "window_end < decision_clock", "panel_audit": activity_audit,
            "base_logit": "logit(clip(return_probability(frozen_model,strict_past_context),1e-5,1-1e-5))",
            "entry": "forced first-entry window excluded; already observed inactive customer windows included"},
        "fit": mixture["fit"], "source_audit": data["audit"], "dynamics_audit": dynamics_audit,
        "split_audit": split_audit, "training": {"seed": int(base.get("seed", 42)),
            "cutoff_at": cutoff.isoformat(), "component_candidates": [COMPONENTS],
            "selected_component_count": COMPONENTS, "regularization_grid": [PENALTY],
            "selection": "fixed first joint prototype; no inner, outer or rollout hyperparameter selection",
            "training_records": len(panel["records"]), "training_sessions": panel["sessions"],
            "training_users": panel["users"], "training_activity_records": len(training_activity),
            "modeled_observation_normalizer": mixture["fit"]["normalizer"]},
        "validation": validation,
        "single_reward_control": {"fit": control["fit"], "validation": control_validation,
            "reward": control["base"].tolist(), "policy": control["statistics"][0][0].tolist(),
            "return_logit_offsets": control["return_offsets"].tolist()},
        "choice_unit": "checkout decision, not purchased unit; inferred END internal; one Bernoulli per nonoverlapping completed 30d activity window",
        "history_clock": "completed session last_action+30m < decision clock; activity window_end < decision clock",
        "runtime_latent": "draw once from source training-prefix profile posterior, else population prior; same private component controls shopping policy and activity offset; never a public KIT39 predictor",
        "method_references": ["https://icml.cc/Conferences/2011/papers/478_icmlpaper.pdf",
            "https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf",
            "https://eml.berkeley.edu/books/choice2nd/Ch06_p134-150.pdf"],
        "limitations": ["Activity head is a supervised observational random effect, not identified long-horizon return reward or causal customer type.",
            "Frozen base GLM was trained under its original Jan29 inclusive-outcome boundary; new activity offsets and personal posteriors use the stricter end<Jan29 boundary.",
            "Jan29 encoder/dynamics/reward/personal bank are refitted; old Feb1 K34 panel likelihood numbers are not directly comparable.",
            "Source action likelihood conditions on observed contexts and schedules; session-count and visit-gap banks are not jointly refitted.",
            "Shared partial-observation transition kernel excludes unidentified unmatched-SKU remove pairs; runtime physical cart dynamics may differ.",
            "Entry/cohort age and conditional timing transport can still alter generated state occupation; no manual activity multiplier is fitted.",
            "Anonymous mixture components are not validated psychological personas; no real-customer experiment or person-level Cosmetics/review source link identifies joint review-shopping reward coefficients."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/models/shopping_profiles_aligned_v6.json"))
    parser.add_argument("--return-model", type=Path, default=Path("data/models/return_model.json"))
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/models/shopping_joint_latent_irl_candidate.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/audit/simulator/joint_latent_shopping_training.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.report.resolve():raise ValueError("model and report paths must differ")
    for path in (args.output, args.report):
        if path.exists():raise FileExistsError(f"immutable fit output exists: {path}")
    base = json.loads(args.base.read_text());return_model = json.loads(args.return_model.read_text())
    if "causal_shopping" in base or "latent_shopping" in base:raise ValueError("joint actor requires the original observed shopping base")
    if base["source"]["files"] != return_model["source"]["files"] or base["source"]["sample_modulus"] != return_model["source"]["sample_modulus"]:
        raise ValueError("shopping and return base must use the identical observed source sample")
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    rows, source = load_prior_source(base, args.source_cache)
    block = fit_joint(rows, base, return_model, shopping.digest(args.return_model))
    block["provenance"] = {"base_path": str(args.base), "base_sha256": shopping.digest(args.base),
        "return_base_path": str(args.return_model), "return_base_sha256": shopping.digest(args.return_model),
        "source_cache": source, "source_cache_sha256": shopping.digest(args.source_cache) if args.source_cache.is_file() else None,
        "fitter_sha256": shopping.digest(__file__), "shopping_fitter_sha256": shopping.digest(shopping.__file__),
        "causal_context_sha256": shopping.digest(shopping.old.__file__),
        "return_fitter_sha256": shopping.digest(Path(__file__).with_name("fit_return.py")),
        "history_helper_sha256": shopping.digest(Path(__file__).with_name("shopping_history.py"))}
    validate_model(block, shopping.digest(args.return_model))
    artifact = deepcopy(base);artifact["latent_shopping"] = block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:json.dump(artifact, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    report = {"passed": True, "model_path": str(args.output), "model_sha256": shopping.digest(args.output),
        **{key: block[key] for key in ("type", "provenance", "training", "fit", "state_count", "mixture_weights",
            "component_offsets", "return_activity", "source_audit", "dynamics_audit", "split_audit", "validation", "single_reward_control", "limitations")},
        "adoption": "candidate; requires generated-sequence fidelity and raw-integrity validation"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"model_path": str(args.output), "model_sha256": report["model_sha256"],
        "state_count": block["state_count"], "weights": block["mixture_weights"],
        "return_offsets": block["return_activity"]["component_logit_offsets"]}), flush=True)


if __name__ == "__main__":main()
