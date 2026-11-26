"""Private session-clock AR(1) return state and mature-window grid fitting.

The likelihood conditions on observed past session counts. It is an approximate
joint window likelihood, not a decision-time prequential forecast or a model of
the session-count process. Persistent customer offsets belong in base_logit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from .source_contract import utc_cutoff


TYPE = "private_session_clock_AR1_return_logit"
HORIZON_DAYS = 30


def make_model(rho, stationary_std, quadrature_order=15):
    model = {"schema_version": 1, "type": TYPE, "rho": float(rho),
             "stationary_std": float(stationary_std), "quadrature_order": quadrature_order,
             "horizon_days": HORIZON_DAYS}
    validate_model(model)
    return model


def validate_model(model):
    if (model.get("schema_version") != 1 or model.get("type") != TYPE
            or model.get("horizon_days") != HORIZON_DAYS):
        raise ValueError("incompatible private return-state model")
    rho, sigma, order = model.get("rho"), model.get("stationary_std"), model.get("quadrature_order")
    if (isinstance(rho, bool) or isinstance(sigma, bool)
            or not isinstance(rho, (int, float)) or not isinstance(sigma, (int, float))
            or not math.isfinite(rho) or not math.isfinite(sigma) or abs(rho) >= 1 or sigma < 0
            or type(order) is not int or order < 3 or order > 61 or order % 2 != 1):
        raise ValueError("return state requires |rho|<1, stationary_std>=0 and an odd quadrature order3..61")


def _count(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("completed session count must be a nonnegative integer")
    return int(value)


def quadrature_grid(model):
    validate_model(model)
    if model["stationary_std"] == 0:
        return np.zeros(1), np.ones(1)
    nodes, weights = hermgauss(model["quadrature_order"])
    return math.sqrt(2) * nodes * model["stationary_std"], weights / math.sqrt(math.pi)


def transition_matrix(model, completed_sessions):
    """Row-normalized GH approximation of the conditional Gaussian kernel."""
    state, weight = quadrature_grid(model)
    n = _count(completed_sessions)
    if n == 0 or len(state) == 1:
        return np.eye(len(state))
    alpha = model["rho"] ** n
    z = state / model["stationary_std"]
    variance = 1 - alpha * alpha
    # GH integrates against the stationary normal: conditional density/prior.
    log_mass = (np.log(weight)[None, :] - .5 * math.log(variance)
                - np.square(z[None, :] - alpha * z[:, None]) / (2 * variance)
                + np.square(z)[None, :] / 2)
    return np.exp(log_mass - logsumexp(log_mass, axis=1, keepdims=True))


def sample_initial_state(model, rng):
    validate_model(model)
    sigma = model["stationary_std"]
    return 0. if sigma == 0 else float(rng.normal(0., sigma))


def advance_state(model, state, rng, completed_sessions=1):
    """Apply exact Gaussian session updates; no session consumes no RNG."""
    validate_model(model)
    if not np.isfinite(state):
        raise ValueError("private return state must be finite")
    n = _count(completed_sessions)
    if n == 0:
        return float(state)
    alpha = model["rho"] ** n
    scale = model["stationary_std"] * math.sqrt(max(0., 1 - alpha * alpha))
    return float(alpha * state) if scale == 0 else float(rng.normal(alpha * state, scale))


def prepare_activity(activity, sessions, *, seed=42, persistent_offsets=None):
    """Bind each anchor only to sessions whose completion is strictly earlier.

    sessions contains one row per actually completed/inferred-known session with
    user_id, session_id, known_at. Tied completion clocks retain their multiplicity.
    Activity labels require full30d followup; missing labels are never zeros.
    """
    required = {"user_id", "at", "target_at", "base_logit", "label"}
    if activity.empty or not required <= set(activity) or not {"user_id", "session_id", "known_at"} <= set(sessions):
        raise ValueError("activity and completed-session observation schemas are required")
    rows, completed = activity.copy(), sessions.copy()
    for table in (rows, completed):
        if not table.user_id.map(lambda uid: isinstance(uid, str) and bool(uid) and uid.strip() == uid).all():
            raise ValueError("return-state user IDs must retain original source text")
    for table, columns in ((rows, ("at", "target_at")), (completed, ("known_at",))):
        for name in columns:
            if not isinstance(table[name].dtype, pd.DatetimeTZDtype) or table[name].isna().any():
                raise ValueError("return-state clocks must be known and timezone-aware")
            table[name] = table[name].dt.tz_convert("UTC").dt.as_unit("ns")
    if "known_at" not in rows:
        rows["known_at"] = rows.target_at
    if (not isinstance(rows.known_at.dtype, pd.DatetimeTZDtype) or rows.known_at.isna().any()
            or rows.known_at.lt(rows.target_at).any()
            or not rows.target_at.eq(rows["at"] + pd.Timedelta(days=HORIZON_DAYS)).all()
            or not rows.label.isin([0, 1]).all() or not pd.api.types.is_numeric_dtype(rows.base_logit)
            or rows.base_logit.isna().any() or not np.isfinite(rows.base_logit).all()):
        raise ValueError("return-state labels need finite logits and fully known30d binary outcomes")
    rows["known_at"] = rows.known_at.dt.tz_convert("UTC").dt.as_unit("ns")
    if completed.session_id.isna().any() or completed.duplicated(["user_id", "session_id"]).any():
        raise ValueError("each completed session must appear exactly once")
    rows = rows.sort_values(["user_id", "at"], kind="stable").reset_index(drop=True)
    previous_end = rows.groupby("user_id", sort=False).target_at.shift()
    if rows.duplicated(["user_id", "at"]).any() or (rows["at"] < previous_end).any():
        raise ValueError("activity windows must be nonoverlapping and unique per user")
    for column, prefix in (("train_user", f"{seed}:"), ("inner_user", f"{seed}:inner:")):
        expected = rows.user_id.map(lambda uid: hashlib.sha256(f"{prefix}{uid}".encode()).digest()[0] < 205)
        if column in rows and (not rows[column].isin([True, False]).all() or not rows[column].eq(expected).all()):
            raise ValueError("return-state user split differs from its frozen hash rule")
        rows[column] = expected
    if persistent_offsets is not None:
        if not isinstance(persistent_offsets, dict) or not set(rows.user_id) <= persistent_offsets.keys():
            raise ValueError("persistent offsets must explicitly cover each source user")
        offset = rows.user_id.map(persistent_offsets)
        if not np.isfinite(offset).all():
            raise ValueError("persistent offsets must be finite")
        rows["base_logit"] = rows.base_logit + offset
    known = {uid: np.sort(group.known_at.astype("int64").to_numpy())
             for uid, group in completed.groupby("user_id", sort=False)}
    before = np.zeros(len(rows), dtype=np.int64)
    for uid, indices in rows.groupby("user_id", sort=False).groups.items():
        before[indices] = np.searchsorted(known.get(uid, np.array([], dtype=np.int64)),
                                          rows.loc[indices, "at"].astype("int64"), side="left")
    rows["completed_sessions_before_at"] = before
    return rows


def _static_particles(prior):
    if (not isinstance(prior, dict) or prior.get("kind") != "action_only_static_return_prior"
            or prior.get("conditioning") != "source_actions_only"
            or prior.get("posterior_uses_activity_labels") is not False
            or not isinstance(prior.get("population_model_fingerprint"), str)
            or len(prior["population_model_fingerprint"]) != 64):
        raise ValueError("static return prior must attest action-only conditioning; activity-conditioned posteriors are forbidden")
    cutoff = utc_cutoff(prior["conditioning_cutoff_at"])
    if prior.get("max_action_known_at") is not None and not utc_cutoff(prior["max_action_known_at"]) < cutoff:
        raise ValueError("static prior contains an action unavailable at its conditioning cutoff")
    offsets, weights = (np.asarray(prior[name], dtype=float) for name in ("offsets", "log_weights"))
    if (offsets.ndim != 1 or not len(offsets) or weights.shape != offsets.shape
            or not np.isfinite(offsets).all() or np.isnan(weights).any() or np.isposinf(weights).any()
            or not np.isfinite(weights).any()):
        raise ValueError("invalid frozen static return particles or log weights")
    return offsets, weights - logsumexp(weights)


def sequence_log_likelihood(prepared, model, *, label_cutoff_at, static_return_prior=None):
    """Joint sequence score with stationary resets, never an as-of forecast score."""
    state, initial = quadrature_grid(model)
    cutoff = utc_cutoff(label_cutoff_at)
    rows = prepared.loc[prepared.known_at.lt(cutoff)]
    total, transitions = 0., {}
    for uid, user in rows.groupby("user_id", sort=False):
        if static_return_prior is None:
            offsets, static_weights = np.zeros(1), np.zeros(1)
        else:
            if uid not in static_return_prior:
                raise ValueError("every scored user needs its frozen action-only static return prior")
            offsets, static_weights = _static_particles(static_return_prior[uid])
        posterior = np.broadcast_to(np.log(initial), (len(offsets), len(initial))).copy()
        likelihood = np.zeros(len(offsets))
        previous = None
        for row in user.itertuples(index=False):
            count = int(row.completed_sessions_before_at)
            if previous is not None:
                n = _count(count - previous)
                if n:
                    if n not in transitions:
                        matrix = transition_matrix(model, n)
                        with np.errstate(divide="ignore"):
                            transitions[n] = np.log(matrix)
                    posterior = logsumexp(posterior[:, :, None] + transitions[n][None], axis=1)
            eta = float(row.base_logit) + offsets[:, None] + state[None, :]
            emission = int(row.label) * eta - np.logaddexp(0., eta)
            normalizer = logsumexp(posterior + emission, axis=1)
            likelihood += normalizer
            posterior = posterior + emission - normalizer[:, None]
            previous = count
        total += float(logsumexp(static_weights + likelihood))
    return {"records": len(rows), "users": int(rows.user_id.nunique()), "log_likelihood": total,
            "negative_log_likelihood_per_observation": -total / len(rows) if len(rows) else None,
            "max_label_known_at": rows.known_at.max().isoformat() if len(rows) else None}


def fit_return_state(activity, sessions, *, inner_cutoff_at, cutoff_at, observation_end_at,
                     seed=42, rhos=(0., .5, .9), stationary_stds=(0., .25, .5, 1.),
                     quadrature_order=15, persistent_offsets=None, static_return_prior=None):
    """Fit rho within each sigma on inner training, select sigma on inner panels.

    Refit rho at the selected sigma on outer training only. Outer user/time panels
    are diagnostics after selection and cannot choose a candidate.
    """
    inner, cutoff, end = map(utc_cutoff, (inner_cutoff_at, cutoff_at, observation_end_at))
    if not inner < cutoff < end:
        raise ValueError("ordered inner/outer/capture cutoffs are required")
    rhos, sigmas = tuple(rhos), tuple(stationary_stds)
    if not rhos or not sigmas or 0. not in sigmas or len(set(rhos)) != len(rhos) or len(set(sigmas)) != len(sigmas):
        raise ValueError("unique parameter grids with the zero-variance candidate are required")
    for rho in rhos:
        for sigma in sigmas:
            make_model(rho, sigma, quadrature_order)
    if static_return_prior is not None and persistent_offsets is not None:
        raise ValueError("static mixture priors and point persistent offsets cannot be added twice")
    rows = prepare_activity(activity, sessions, seed=seed, persistent_offsets=persistent_offsets)
    prior_fingerprint = None
    if static_return_prior is not None:
        if not isinstance(static_return_prior, dict) or not set(rows.user_id) <= static_return_prior.keys():
            raise ValueError("frozen action-only static priors must cover every activity user")
        fingerprints = set()
        for uid in rows.user_id.unique():
            prior = static_return_prior[uid]
            _static_particles(prior)
            if utc_cutoff(prior["conditioning_cutoff_at"]) > cutoff:
                raise ValueError("static action conditioning exceeds the fitted cutoff")
            fingerprints.add(prior["population_model_fingerprint"])
        if len(fingerprints) != 1:
            raise ValueError("all dynamic-state candidates must use the same frozen population GMM")
        prior_fingerprint = fingerprints.pop()
    if rows.known_at.gt(end).any():
        raise ValueError("activity outcomes exceed their declared observation capture")
    early, later = rows.known_at.lt(inner), rows["at"].ge(inner) & rows.known_at.lt(cutoff)
    outer_train = rows.train_user & rows.known_at.lt(cutoff)
    inner_train = rows.train_user & rows.inner_user & early
    validation_masks = {"user_holdout": rows.train_user & ~rows.inner_user & early,
                        "time_holdout": rows.train_user & rows.inner_user & later,
                        "user_and_time_holdout": rows.train_user & ~rows.inner_user & later}
    if not inner_train.any() or not outer_train.any() or not any(mask.any() for mask in validation_masks.values()):
        raise ValueError("return-state fitting needs inner training, inner validation and outer training support")

    def score(mask, model, bound):
        return sequence_log_likelihood(rows.loc[mask], model, label_cutoff_at=bound,
                                       static_return_prior=static_return_prior)

    def fit_rho(sigma, mask, bound):
        candidates = []
        for rho in ((0.,) if sigma == 0 else rhos):
            model = make_model(rho, sigma, quadrature_order)
            candidates.append((score(mask, model, bound)["negative_log_likelihood_per_observation"], abs(rho), rho, model))
        return min(candidates, key=lambda item: item[:3])

    trials = []
    for sigma in sigmas:
        train_nll, _, _, candidate = fit_rho(sigma, inner_train, inner)
        validation = {name: score(mask, candidate, cutoff) for name, mask in validation_masks.items()}
        observations = sum(value["records"] for value in validation.values())
        validation_nll = -sum(value["log_likelihood"] for value in validation.values()) / observations
        trials.append({"stationary_std": sigma, "inner_fitted_rho": candidate["rho"],
                       "inner_train_nll": train_nll, "inner_validation_nll": validation_nll,
                       "inner_validation": validation})
    selected = min(trials, key=lambda row: (row["inner_validation_nll"], row["stationary_std"], abs(row["inner_fitted_rho"])))
    _, _, _, model = fit_rho(selected["stationary_std"], outer_train, cutoff)
    diagnostics = {"train": outer_train,
                   "user_holdout": ~rows.train_user & rows.known_at.lt(cutoff),
                   "time_holdout": rows.train_user & rows["at"].ge(cutoff),
                   "user_and_time_holdout": ~rows.train_user & rows["at"].ge(cutoff)}
    model.update(private_state=True, transition_unit="actually completed session",
        initial_distribution="stationary N(0,stationary_std^2)", no_session_transition="identity; no RNG draw",
        emission="Bernoulli(sigmoid(base_logit + private_eta)); persistent offset already included or explicitly added once",
        rho_identified=False if model["stationary_std"] == 0 else None,
        rho_grid_estimated=model["stationary_std"] > 0,
        fit={"seed": seed, "inner_cutoff_at": inner.isoformat(), "cutoff_at": cutoff.isoformat(),
             "observation_end_at": end.isoformat(), "label_available_rule": "known_at < label_cutoff",
             "session_available_rule": "session_completion_known_at < current anchor; n is difference of past counts",
             "user_rule": "sha256(seed:user_id)[0]<205; inner sha256(seed:inner:user_id)[0]<205",
             "selection": "rho fitted on inner training for each sigma; sigma chosen on inner panels; rho refitted on outer training; no outer diagnostic selection",
             "rhos": list(rhos), "stationary_stds": list(sigmas), "trials": trials,
             "persistent_offset_supplied_separately": persistent_offsets is not None,
             "static_return_prior": None if static_return_prior is None else {
                 "population_model_fingerprint": prior_fingerprint,
                 "conditioning": "source_actions_only; no individual activity-label posterior",
                 "base_logit_contract": "global GLM only; static offsets integrated once",
                 "scope": "descriptive fixed-prefix action conditioning; no as-of forecast prior claim",
                 "integration": "one fixed static particle per user sequence, independent GH dynamic filters, then logsumexp"},
             "outer_diagnostics": {name: score(mask, model, end) for name, mask in diagnostics.items()}},
        limitations=["Conditional observational window likelihood; session-count endogeneity and persistent-offset errors can confound rho and sigma.",
            "GH finite-state transition rows are a numerical approximation; inspect quadrature sensitivity before adoption.",
            "Each scored user/panel resets to the stationary prior; joint likelihood may include a previous label unavailable at the next anchor, so it is not a decision-time prequential forecast.",
            "No independent NEW/late lifecycle actor, permanent churn state or beyond-capture extrapolation is identified.",
            "rho is unidentified at zero stationary variance; zero is retained without artificial jitter.",
            "No public feature or missingness predictor is added; live engine integration is a separate step."])
    if static_return_prior is not None:
        model["emission"] = "Bernoulli(sigmoid(global_GLM_logit + fixed_static_theta_return + private_eta)); action-only static mixture integrated once per user sequence"
        model["limitations"].append("Action-only describes individual posterior conditioning; the frozen population GMM may already use training activity labels, so this is conditional dynamic-state calibration, not a joint or independent population refit.")
    return model


def save_model(path, model):
    validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(model, stream, ensure_ascii=False, indent=2, allow_nan=False)
