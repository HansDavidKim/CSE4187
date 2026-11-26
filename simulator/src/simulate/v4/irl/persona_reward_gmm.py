"""Joint Gaussian offsets conditional on frozen source-prefix BiGRU personas.

Persona labels and their population prior are fixed. QMC EM integrates missing
individual shopping/activity evidence; it never substitutes a zero MAP target.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp

from . import fit_latent_shopping as shopping
from .fit_gaussian_rewards import PARAMETERS, parameter_indices
from .fit_joint_reward_gmm import INITIAL_SD, EIGENVALUE_FLOOR, policy_logs, sobol_normal
from .source_contract import canonical_available_actions, utc_cutoff


TYPE = "source_persona_joint_reward_gmm"
DAILY_TYPE = "source_persona_daily_reward_gmm"
STATIC_PRIOR_TYPE = "action_only_static_return_prior"


def _clock(frame, name):
    if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or frame[name].isna().any():
        raise ValueError("persona GMM observation clocks must be known and timezone-aware")
    return frame[name].dt.tz_convert("UTC")


def _users(frame):
    if not frame.user_id.map(lambda uid: isinstance(uid, str) and bool(uid) and uid.strip() == uid).all():
        raise ValueError("persona GMM requires unchanged original user ID text")


def _panel(actions, activity, users, states):
    """The union keeps return-only users as zero sparse action rows."""
    a, r = actions.loc[actions.user_id.isin(users)], activity.loc[activity.user_id.isin(users)]
    observed = sorted(set(a.loc[a["count"].gt(0), "user_id"]) | set(r.user_id))
    index = {uid: i for i, uid in enumerate(observed)}
    counts = (sparse.coo_matrix((a["count"], (a.user_id.map(index), a.state * 5 + a.action)),
                               shape=(len(observed), states * 5)).tocsr()
              if len(a) else sparse.csr_matrix((len(observed), states * 5)))
    incidence = (sparse.coo_matrix((np.ones(len(r)), (r.user_id.map(index), np.arange(len(r)))),
                                  shape=(len(observed), len(r))).tocsr()
                 if len(r) else sparse.csr_matrix((len(observed), 0)))
    if not len(observed) or counts.sum() + len(r) == 0:
        raise ValueError("each persona needs observed optimization and internal-selection support")
    return {"users": observed, "counts": counts, "incidence": incidence, "activity": r,
            "observations": float(counts.sum() + len(r)),
            "return_only_users": int(((np.asarray(counts.sum(axis=1)).ravel() == 0)
                                      & (np.asarray(incidence.sum(axis=1)).ravel() > 0)).sum())}


def _likelihood(panel, sample, base, mask, transition, *, policy_log_provider=None, actor_index=None):
    if panel["counts"].nnz:
        logs = (policy_logs(base, mask, transition, sample) if policy_log_provider is None
                else np.asarray(policy_log_provider(actor_index, sample), dtype=float))
        if logs.shape != (len(sample), mask.size) or not np.isfinite(logs[:, mask.ravel()]).all():
            raise ValueError("policy log provider must return finite sample-by-state/action probabilities")
        if policy_log_provider is not None:
            probability = np.where(mask[None], np.exp(logs.reshape(len(sample), *mask.shape)), 0.)
            if not np.allclose(probability.sum(axis=2), 1., atol=1e-8, rtol=0):
                raise ValueError("policy log provider must preserve normalized continuing outcomes")
            logs = np.where(mask.ravel()[None], logs, 0.)
        values = panel["counts"] @ logs.T
    else:
        values = np.zeros((len(panel["users"]), len(sample)))
    activity = panel["activity"]
    if len(activity):
        eta = activity.base_logit.to_numpy()[:, None] + sample[None, :, 4]
        values += panel["incidence"] @ (activity.label.to_numpy()[:, None] * eta - np.logaddexp(0., eta))
    evidence = logsumexp(values, axis=1) - np.log(len(sample))
    posterior = np.exp(values - logsumexp(values, axis=1, keepdims=True)).sum(axis=0)
    return float(-evidence.sum() / panel["observations"]), posterior


def _sample(mean, covariance, normal, active):
    result = np.zeros((len(normal), 5))
    result[:, active] = mean[active] + normal @ np.linalg.cholesky(covariance[np.ix_(active, active)]).T
    return result


def _moments(sample, posterior, active, initial_sd=INITIAL_SD):
    weight = posterior / posterior.sum()
    values = sample[:, active]
    mean = weight @ values
    centered = values - mean
    covariance = (centered.T * weight) @ centered
    eigenvalue, vector = np.linalg.eigh(covariance)
    covariance = (vector * np.maximum(eigenvalue, EIGENVALUE_FLOOR)) @ vector.T
    embedded_mean, embedded_covariance = np.zeros(5), np.diag(np.square(initial_sd))
    embedded_mean[active] = mean
    embedded_covariance[np.ix_(active, active)] = covariance
    return embedded_mean, embedded_covariance


def action_only_static_return_prior(latent, block, actions, assignments, *, cutoff_at,
                                    qmc_power=7, batch_users=256):
    """Condition frozen persona Gaussian particles on shopping choices only.

    Activity labels and saved full-history profile posteriors are never read.
    Shared persona draws and bounded sparse batches avoid a user-by-state-by-Q
    tensor. Whole-prefix action conditioning is descriptive, not an as-of prior.
    """
    from .reward_gmm import validate_gmm
    cutoff = utc_cutoff(cutoff_at)
    if (block.get("type") != TYPE or block.get("backend", "retail_session") != "retail_session"
            or type(qmc_power) is not int or not 1 <= qmc_power <= 14
            or type(batch_users) is not int or batch_users < 1):
        raise ValueError("action-only return priors require a frozen retail persona GMM and bounded QMC batches")
    available = canonical_available_actions(latent.get("encoder", {}).get("available_actions"))
    validate_gmm(block, block.get("base_model_sha256"), block.get("return_model_sha256"), available_actions=available)
    fingerprint = hashlib.sha256(json.dumps(block, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    frozen = assignments.copy()
    if (not {"user_id", "cluster", "assignment_cutoff", "representation_end_at"} <= set(frozen)
            or frozen.user_id.duplicated().any() or frozen.cluster.isna().any()):
        raise ValueError("action-only priors require unique frozen persona assignments")
    _users(frozen)
    if (_clock(frozen, "assignment_cutoff").gt(cutoff).any()
            or _clock(frozen, "representation_end_at").ge(_clock(frozen, "assignment_cutoff")).any()):
        raise ValueError("action-only prior persona assignment exceeds the frozen prefix")
    frozen["cluster"] = frozen.cluster.astype(str)
    labels = block["persona_labels"]
    if not set(frozen.cluster) <= set(labels):
        raise ValueError("action-only prior personas differ from the frozen Gaussian groups")
    a = actions.copy()
    if not {"user_id", "state", "action", "known_at"} <= set(a):
        raise ValueError("action-only priors accept sparse shopping observations, not activity posteriors")
    _users(a)
    base = np.asarray(latent["common_reward"], dtype=float)
    mask = np.asarray(latent["action_mask"], dtype=bool)
    if len(a):
        a["known_at"] = _clock(a, "known_at")
        if (not pd.api.types.is_integer_dtype(a.state) or not pd.api.types.is_integer_dtype(a.action)
                or a.state.lt(0).any() or a.state.ge(len(base)).any() or not a.action.isin(range(5)).all()
                or not mask[a.state.to_numpy(), a.action.to_numpy()].all()):
            raise ValueError("action-only prior observations violate actor support")
        a = a.loc[a.known_at.lt(cutoff)]
    a["count"] = a["count"] if "count" in a else 1.
    if not np.isfinite(a["count"]).all() or a["count"].lt(0).any():
        raise ValueError("action-only counts must be finite and nonnegative")
    deviations = np.asarray(latent.get("component_reward_deviations", np.zeros((len(latent["component_offsets"]), *base.shape))))
    transition = shopping.DISCOUNT * sparse.vstack(shopping.unpack_matrices(latent), format="csr")
    active, result = parameter_indices(available), {}
    for component, label in enumerate(labels):
        actor_index = block["component_model_indices"][component]
        users = frozen.loc[frozen.cluster.eq(label), "user_id"].tolist()
        normal = sobol_normal(qmc_power, 1709 + actor_index, dimension=len(active))
        sample = _sample(np.asarray(block["means"][component]), np.asarray(block["covariances"][component]), normal, active)
        scale = block.get("dispersion_scales", [1.] * len(labels))[component]
        mean = np.asarray(block["means"][component])
        sample[:, active] = mean[active] + scale * (sample[:, active] - mean[active])
        temperature = block.get("choice_temperatures", [1.] * len(labels))[component]
        choice_offsets = sample.copy(); choice_offsets[:, :4] /= temperature
        logs = policy_logs((base + deviations[actor_index]) / temperature, mask, transition, choice_offsets)
        offsets = sample[:, 4].copy()
        offsets.setflags(write=False)
        group = a.loc[a.user_id.isin(users)]
        for start in range(0, len(users), batch_users):
            chunk = users[start:start + batch_users]
            index = {uid: i for i, uid in enumerate(chunk)}
            part = group.loc[group.user_id.isin(chunk)]
            counts = (sparse.coo_matrix((part["count"], (part.user_id.map(index), part.state * 5 + part.action)),
                                       shape=(len(chunk), len(base) * 5)).tocsr()
                      if len(part) else sparse.csr_matrix((len(chunk), len(base) * 5)))
            weights = counts @ logs.T
            weights -= logsumexp(weights, axis=1, keepdims=True)
            maxima = part.groupby("user_id").known_at.max() if len(part) else {}
            for position, uid in enumerate(chunk):
                log_weights = weights[position].copy()
                log_weights.setflags(write=False)
                result[uid] = {"kind": STATIC_PRIOR_TYPE, "conditioning": "source_actions_only",
                    "posterior_uses_activity_labels": False, "population_model_fingerprint": fingerprint,
                    "conditioning_cutoff_at": cutoff.isoformat(), "max_action_known_at": maxima[uid].isoformat() if uid in maxima else None,
                    "scope": "descriptive whole frozen-prefix shopping-choice conditioning; not an as-of forecast prior",
                    "offsets": offsets, "log_weights": log_weights}
    return result


def fit_persona_reward_gmm(latent, actions, activity, assignments, *, component_model_indices,
                          cutoff_at, seed=42, max_updates=8, qmc_power=7,
                          condition_on_profile_components=False,
                          base_model_sha256=None, return_model_sha256=None,
                          policy_log_provider=None, active_parameter_indices=None,
                          backend="retail_session", initial_sd=INITIAL_SD):
    """Fit known-persona offsets using pre-cutoff inner users only for selection.

    actions: user_id,state,action,known_at and optional observed count (default1).
    activity: user_id,at,target_at,base_logit,label and optional known_at.
    assignments: user_id,cluster,assignment_cutoff,representation_end_at.
    base_logit is the frozen global return GLM without a persona return offset.
    latent is the compiled latent_shopping block, not its enclosing artifact.
    For tianchi_daily, latent is a state_count/component_count/action_mask
    descriptor, activity is empty, and a daily joint-outcome provider is required.
    """
    cutoff = utc_cutoff(cutoff_at)
    initial_sd = np.asarray(initial_sd, dtype=float)
    if initial_sd.shape != (5,) or not np.isfinite(initial_sd).all() or (initial_sd <= 0).any():
        raise ValueError("canonical5 initialization scales must be finite and positive")
    if (type(max_updates) is not int or max_updates < 0 or type(qmc_power) is not int
            or not 1 <= qmc_power <= 14 or type(condition_on_profile_components) is not bool):
        raise ValueError("valid fixed QMC/iteration budgets and conditioning flag required")
    if backend not in {"retail_session", "tianchi_daily"} or policy_log_provider is not None and not callable(policy_log_provider):
        raise ValueError("an explicit supported backend and callable policy log provider are required")
    mask = np.asarray(latent["action_mask"], dtype=bool)
    if backend == "tianchi_daily":
        states, actor_count = latent.get("state_count"), latent.get("component_count")
        provided_active = () if active_parameter_indices is None else tuple(active_parameter_indices)
        if (type(states) is not int or states < 1 or type(actor_count) is not int or actor_count < 2
                or mask.shape != (states, 5) or not mask[:, :4].all() or mask[:, 4].any()
                or policy_log_provider is None or provided_active != (0, 3, 4)):
            raise ValueError("daily GMM needs a four-outcome/no-END descriptor, provider and active(view,purchase,return)")
        base = deviations = None
        available, names = canonical_available_actions(("view", "purchase")), [*PARAMETERS[:4], "return"]
    else:
        base = np.asarray(latent["common_reward"], dtype=float)
        offsets = np.asarray(latent["component_offsets"], dtype=float)
        deviations = np.asarray(latent.get("component_reward_deviations", np.zeros((len(offsets), *base.shape))), dtype=float)
        available, names = canonical_available_actions(latent.get("encoder", {}).get("available_actions")), PARAMETERS
        states, actor_count = len(base), len(offsets)
        if (base.ndim != 2 or base.shape[1] != 4 or offsets.ndim != 2 or offsets.shape[1] != 4
                or offsets.any() or mask.shape != (states, 5) or deviations.shape != (actor_count, *base.shape)
                or not np.isfinite(base).all() or not np.isfinite(deviations).all()
                or latent.get("discount", shopping.DISCOUNT) != shopping.DISCOUNT
                or active_parameter_indices is not None and list(active_parameter_indices) != parameter_indices(available)):
            raise ValueError("persona GMM requires aligned group reward deviations and zero nominal action offsets")
    if not isinstance(component_model_indices, dict):
        raise ValueError("a frozen persona-label to actor-index mapping is required")
    mapping = {str(label): index for label, index in component_model_indices.items()}
    if (len(mapping) != len(component_model_indices) or len(mapping) < 2 or len(set(mapping.values())) != len(mapping)
            or any(type(index) is not int or not 0 <= index < actor_count for index in mapping.values())):
        raise ValueError("frozen persona labels must map to valid actor indices")
    labels = sorted(mapping)
    if condition_on_profile_components and (len(labels) != actor_count or sorted(mapping.values()) != list(range(actor_count))):
        raise ValueError("profile conditioning requires one-to-one persona/actor groups")
    if not {"user_id", "cluster", "assignment_cutoff", "representation_end_at"} <= set(assignments) or assignments.empty:
        raise ValueError("frozen training-prefix persona assignments required")
    frozen = assignments.copy()
    _users(frozen)
    if frozen.user_id.duplicated().any() or frozen.cluster.isna().any():
        raise ValueError("frozen persona assignments must name unique known users")
    frozen["cluster"] = frozen.cluster.astype(str)
    assigned_at, prefix_end = (_clock(frozen, name) for name in ("assignment_cutoff", "representation_end_at"))
    if prefix_end.ge(assigned_at).any() or assigned_at.gt(cutoff).any() or not set(frozen.cluster) <= set(labels):
        raise ValueError("persona assignments exceed their frozen prefix or actor vocabulary")
    outer = frozen.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
    inner = frozen.user_id.map(lambda uid: hashlib.sha256(f"{seed}:inner:{uid}".encode()).digest()[0] < 205)
    a, r = actions.copy(), activity.copy()
    if backend == "tianchi_daily" and r.empty:
        r = pd.DataFrame(columns=["user_id", "at", "target_at", "base_logit", "label"])
    if not {"user_id", "state", "action", "known_at"} <= set(a) or not {"user_id", "at", "target_at", "base_logit", "label"} <= set(r):
        raise ValueError("aligned sparse action observations and complete activity offsets required")
    for frame in (a, r):
        _users(frame)
    if backend == "tianchi_daily" and len(r) or backend == "retail_session" and not len(r):
        raise ValueError("daily return evidence belongs to daily outcomes; retail requires complete30d activity observations")
    if len(a):
        a["known_at"] = _clock(a, "known_at")
        if (not pd.api.types.is_integer_dtype(a.state) or not pd.api.types.is_integer_dtype(a.action)
                or a.state.lt(0).any() or a.state.ge(states).any() or not a.action.isin(range(5)).all()
                or not mask[a.state.to_numpy(), a.action.to_numpy()].all()):
            raise ValueError("persona action observations violate fitted state/action support")
    a["count"] = a["count"] if "count" in a else 1.
    if not np.isfinite(a["count"]).all() or a["count"].lt(0).any():
        raise ValueError("observed action counts must be finite and nonnegative")
    if len(r):
        for name in ("at", "target_at"):
            r[name] = _clock(r, name)
        r["known_at"] = _clock(r, "known_at") if "known_at" in r else r.target_at
        if (not r.label.isin([0, 1]).all() or not np.isfinite(r.base_logit).all()
                or not r.target_at.eq(r["at"] + pd.Timedelta(days=30)).all()
                or r.known_at.lt(r.target_at).any() or r.duplicated(["user_id", "at"]).any()):
            raise ValueError("activity needs complete30d binary outcomes and fixed finite GLM offsets")
        r = r.sort_values(["user_id", "at"], kind="stable")
        if r["at"].lt(r.groupby("user_id", sort=False).target_at.shift()).any():
            raise ValueError("persona activity windows must not overlap")
        if "train_user" in r:
            expected = r.user_id.map(lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).digest()[0] < 205)
            if not r.train_user.eq(expected).all():
                raise ValueError("activity outer split differs from the frozen user hash")
    else:
        r["known_at"] = pd.Series([], dtype="datetime64[ns, UTC]")
    a = a.loc[a.known_at.lt(cutoff) & a["count"].gt(0)] if len(a) else a
    r = r.loc[r.known_at.lt(cutoff)]
    transition = (shopping.DISCOUNT * sparse.vstack(shopping.unpack_matrices(latent), format="csr")
                  if policy_log_provider is None else None)
    means, covariances, priors, reports = [], [], [], {}
    for label in labels:
        group = frozen.cluster.eq(label)
        total = int((group & outer).sum())
        if not total:
            raise ValueError("each actor needs a nonzero source training-prefix persona prior")
        priors.append(total)
        train = _panel(a, r, set(frozen.loc[group & outer & inner, "user_id"]), states)
        selection = _panel(a, r, set(frozen.loc[group & outer & ~inner, "user_id"]), states)
        active = (list(active_parameter_indices) if backend == "tianchi_daily" else
                  [index for index in parameter_indices(available) if index < 4] if train["counts"].sum() else [])
        if backend == "retail_session" and len(train["activity"]):
            active.append(4)
        integration = parameter_indices(available)
        normal = sobol_normal(qmc_power, 293 + mapping[label], dimension=len(integration))
        actor_base = base + deviations[mapping[label]] if base is not None else None
        mean, covariance = np.zeros(5), np.diag(np.square(initial_sd))
        best, trace = None, []
        for step in range(max_updates + 1):
            sample = _sample(mean, covariance, normal, integration)
            train_nll, posterior = _likelihood(train, sample, actor_base, mask, transition,
                policy_log_provider=policy_log_provider, actor_index=mapping[label])
            selection_nll, _ = _likelihood(selection, sample, actor_base, mask, transition,
                policy_log_provider=policy_log_provider, actor_index=mapping[label])
            trace.append({"iteration": step, "train_nll": train_nll, "internal_selection_nll": selection_nll})
            if best is None or selection_nll < best[0]:
                best = selection_nll, step, mean.copy(), covariance.copy()
            if step < max_updates:
                mean, covariance = _moments(sample, posterior, active, initial_sd)
        means.append(best[2].tolist()); covariances.append(best[3].tolist())
        reports[label] = {"selected_iteration": best[1], "trace": trace,
            "fitted_parameter_names": [names[index] for index in active],
            "training_users": len(train["users"]), "selection_users": len(selection["users"]),
            "training_return_only_users": train["return_only_users"],
            "selection_return_only_users": selection["return_only_users"],
            "source_training_prefix_users_for_prior": total}
    prior = np.asarray(priors, dtype=float); prior /= prior.sum()
    result = {"schema_version": 1, "type": TYPE if backend == "retail_session" else DAILY_TYPE,
        "backend": backend, "purpose": "synthesis_only", "parameter_names": list(names),
        "component_count": len(labels), "persona_labels": labels,
        "component_fitted_parameter_names": [reports[label]["fitted_parameter_names"] for label in labels],
        "component_model_indices": [mapping[label] for label in labels],
        "condition_on_profile_components": condition_on_profile_components,
        "mixture_weights": prior.tolist(), "means": means, "covariances": covariances,
        "fit": {"seed": seed, "cutoff_at": cutoff.isoformat(), "max_updates": max_updates,
            "points_per_persona": 2 ** qmc_power, "persona_groups_refitted_or_reclustered": False,
            "qmc_seed_by_actor": "293 + component_model_index",
            "covariance_eigenvalue_floor": EIGENVALUE_FLOOR, "initial_covariance_sd": initial_sd.tolist(),
            "unidentified_coordinate_embedding": "unused zero mean and fixed independent reference variance; never a MAP observation",
            "pre_cutoff_observed_users_without_frozen_assignment": len((set(a.user_id) | set(r.user_id)) - set(frozen.user_id)),
            "population_prior": "all frozen source outer-training-prefix persona assignments, including zero-emission customers",
            "split": "outertrain AND innertrain optimize; outertrain AND NOT innertrain select; all known_at strictly before cutoff",
            "outer_holdout_or_postcutoff_used_for_selection": False, "personas": reports,
            "return_coordinate": ("offset added to frozen global GLM; not an increment on a preexisting group return offset"
                                  if backend == "retail_session" else "daily activity reward inside four joint outcomes; not a30d GLM logit")},
        "limitations": ["Conditional calibration: frozen persona actors/base GLM can already use internal selection users.",
            "Frozen prefix groups identify descriptive source strata, not psychological types or individual causal effects.",
            "QMC EM is approximate; covariance eigenvalue flooring affects raw dispersion, preserved separately from runtime scales.",
            "No-emission users affect the registered persona prior only; missing shopping/activity is integrated, never a zero MAP observation.",
            "Coordinates with no group training channel retain an unused reference marginal and are not identified."]}
    if available != canonical_available_actions():
        result.update(available_actions=list(available), fitted_parameter_names=[names[i] for i in parameter_indices(available)])
    if base_model_sha256 is not None:
        result["base_model_sha256"] = base_model_sha256
    if return_model_sha256 is not None:
        result["return_model_sha256"] = return_model_sha256
    return result
