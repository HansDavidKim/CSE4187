"""Source-fitted continuous customer rewards around frozen v4 IRL dynamics.

This two-stage synthesis model fits individual MAP reward offsets and conditional
Gaussian moments. Frozen discrete donor posteriors remain the component gate;
the parent's forecast validation does not validate this Gaussian approximation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from simulate.v4.irl import fit_joint_latent_shopping as joint
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl import source_phases
from simulate.v4.irl.source_contract import canonical_available_actions, source_profile_id


TYPE = "source_map_gaussian_reward_offsets"
PARAMETERS = ["view", "cart", "remove", "purchase", "return_logit"]
PENALTIES = (.5, 2., 10.)
FIT_SCOPE = "all_fully_observed_source_user_map"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_indices(available_actions=None):
    return [PARAMETERS.index(action) for action in canonical_available_actions(available_actions)] + [4]


def validate_parameter_support(block, available_actions=None):
    available = canonical_available_actions(block.get("available_actions"))
    if available_actions is not None and available != canonical_available_actions(available_actions):
        raise ValueError("continuous reward observation support differs from its fitted actor")
    if "available_actions" in block and block["available_actions"] != list(available):
        raise ValueError("continuous observation support must use canonical order")
    if (available != tuple(PARAMETERS[:4]) or "fitted_parameter_names" in block) and block.get("fitted_parameter_names") != [PARAMETERS[i] for i in parameter_indices(available)]:
        raise ValueError("continuous reward must declare its identified parameter marginal")
    return available


def validate_gaussian(block, base_sha256, return_sha256, *, available_actions=None):
    """Reject stale source bindings, invalid dispersions and forecast-use claims."""
    if (not isinstance(block, dict) or type(block.get("schema_version")) is not int
            or block.get("schema_version") != 1 or block.get("type") != TYPE
            or block.get("purpose") != "synthesis_only"
            or block.get("parameter_names") != PARAMETERS):
        raise ValueError("incompatible Gaussian reward sidecar")
    for name, expected in (("base_model_sha256", base_sha256), ("return_model_sha256", return_sha256)):
        value = block.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value) or value != expected):
            raise ValueError(f"Gaussian reward sidecar requires its exact frozen {name}")
    fit = block.get("fit")
    if (not isinstance(fit, dict) or fit.get("scope") != FIT_SCOPE
            or fit.get("forecast_validation_uses_gaussian") is not False):
        raise ValueError("Gaussian reward fit must remain synthesis only")
    try:
        means = np.asarray(block.get("means"), dtype=float)
        variances = np.asarray(block.get("variances"), dtype=float)
        weights = np.asarray(block.get("mixture_weights"), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid Gaussian reward arrays") from exc
    count = block.get("component_count")
    if (type(count) is not int or count not in (2, 3, 4)
            or means.shape != (count, 5) or variances.shape != means.shape
            or weights.shape != (count,) or not np.isfinite(means).all()
            or not np.isfinite(variances).all() or (variances < 0).any()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1)):
        raise ValueError("invalid Gaussian reward means, variances or component weights")
    validate_parameter_support(block, available_actions)


def sample_reward_policy(model, block, component_id, rng, matrices=None):
    """Draw once per customer, then solve that customer's frozen-dynamics policy."""
    latent = model.get("latent_shopping", model)
    available = canonical_available_actions(latent.get("encoder", {}).get("available_actions"))
    validate_parameter_support(block, available)
    active = parameter_indices(available)
    if (isinstance(component_id, (bool, np.bool_)) or not isinstance(component_id, (int, np.integer))
            or not 0 <= component_id < len(latent["mixture_weights"])):
        raise ValueError("Gaussian component must be the persistent customer draw")
    mean = np.asarray(block["means"][component_id], dtype=float)
    variance = np.asarray(block["variances"][component_id], dtype=float)
    if mean.shape != (5,) or variance.shape != (5,) or not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("invalid Gaussian reward draw parameters")
    # A point mass preserves both the original policy and its RNG stream.
    if len(active) == 5:
        theta = mean.copy() if not variance.any() else rng.normal(mean, np.sqrt(variance))
    else:
        theta = np.zeros(5)
        theta[active] = mean[active] if not variance[active].any() else rng.normal(mean[active], np.sqrt(variance[active]))
    action_active = [i for i in active if i < 4]
    if not variance[action_active].any() and np.array_equal(theta[action_active], np.asarray(latent["component_offsets"][component_id])[action_active]):
        probability = np.asarray(latent["component_policies"][component_id], dtype=float).copy()
    else:
        matrices = shopping.unpack_matrices(latent) if matrices is None else matrices
        probability, _, _ = shopping.soft_policy(
            shopping._reward(np.asarray(latent["common_reward"]) + theta[:4]),
            matrices, np.asarray(latent["action_mask"], dtype=bool))
    probability[~np.asarray(latent["action_mask"], dtype=bool)] = 0.
    return {"action_offsets": theta[:4], "return_offset": float(theta[4]), "policy": probability}


def map_objective(theta, counts, activity, base, matrices, mask, centre, penalty, *, available_actions=None):
    """Exact identified-coordinate MAP gradient through the soft Bellman solution."""
    active = parameter_indices(available_actions)
    theta = np.asarray(theta)
    if len(active) != 5:
        theta = theta.copy()
        theta[[i for i in range(5) if i not in active]] = 0.
    if np.asarray(counts)[~mask].any():
        raise ValueError("Gaussian likelihood contains an unsupported action")
    reward = shopping._reward(base + theta[:4])
    likelihood, gradient, _ = shopping.policy_likelihood_gradient(reward, counts, matrices, mask)
    eta = activity.base_logit.to_numpy() + theta[4]
    labels = activity.label.to_numpy()
    difference = theta - centre
    if len(active) != 5:
        difference[[i for i in range(5) if i not in active]] = 0.
    loss = -likelihood - (labels * eta - np.logaddexp(0, eta)).sum()
    derivative = np.r_[-gradient[:, :4].sum(axis=0), (expit(eta) - labels).sum()]
    if len(active) != 5:
        derivative[[i for i in range(5) if i not in active]] = 0.
    return float(loss + penalty * difference @ difference / 2), derivative + penalty * difference


def fit_user_offsets(counts, activity, base, matrices, mask, centre, penalty, *, available_actions=None):
    active = parameter_indices(available_actions)
    if len(active) == 5:
        objective = lambda theta: map_objective(theta, counts, activity, base, matrices, mask, centre, penalty)
    else:
        def objective(value):
            theta = np.zeros(5)
            theta[active] = value
            loss, gradient = map_objective(theta, counts, activity, base, matrices, mask, centre, penalty,
                                          available_actions=available_actions)
            return loss, gradient[active]
    result = minimize(objective, np.asarray(centre, dtype=float)[active].copy(), jac=True,
        method="L-BFGS-B", options={"maxiter": 250, "gtol": 1e-6, "ftol": 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"individual Gaussian reward MAP did not converge: {result.message}")
    theta = np.zeros(5)
    theta[active] = result.x
    return theta, {"iterations": int(result.nit), "gradient_max_abs": float(np.abs(result.jac).max())}


def source_session_masks(sessions, observation_end, cutoff=joint.CUTOFF):
    """Keep complete sessions together; purge sessions crossing the tuning clock."""
    valid = (~sessions.invalid_total_cart & ~sessions.ambiguous_clock_sessions & ~sessions.is_censored
             & sessions.terminal_known_at.le(pd.Timestamp(observation_end)))
    cutoff = pd.Timestamp(cutoff)
    return {"all": valid, "prefix": valid & sessions.terminal_known_at.lt(cutoff),
            "later": valid & sessions.start_at.ge(cutoff)}


def weighted_gaussian_moments(coefficients, responsibility, original_means, *, available_actions=None):
    """Empirical conditional moments; no artificial within-component jitter."""
    values = np.asarray(coefficients, dtype=float)
    responsibility = np.asarray(responsibility, dtype=float)
    means = np.asarray(original_means, dtype=float).copy()
    active = parameter_indices(available_actions)
    if (values.ndim != 2 or values.shape[1] != 5
            or responsibility.shape != (len(values), len(means)) or means.shape[1:] != (5,)
            or not np.isfinite(values[:, active]).all() or not np.isfinite(means).all()
            or not np.isfinite(responsibility).all() or (responsibility < 0).any()
            or not np.allclose(responsibility.sum(axis=1), 1)):
        raise ValueError("invalid MAP values or frozen component responsibilities")
    masses = responsibility.sum(axis=0)
    variances = np.zeros_like(means)
    for component, mass in enumerate(masses):
        if mass >= 1e-6:
            weights = responsibility[:, component] / mass
            if len(active) == 5:
                means[component] = weights @ values
                variances[component] = weights @ np.square(values - means[component])
            else:
                means[component, active] = weights @ values[:, active]
                variances[component, active] = weights @ np.square(values[:, active] - means[component, active])
    if len(active) != 5:
        means[:, [i for i in range(5) if i not in active]] = 0.
    return means, variances, masses


def _observations(data, states, selected_sessions, activity, users, state_count):
    indices = [np.arange(row.record_start, row.record_end)
               for row in data["sessions"].loc[selected_sessions].itertuples()]
    indices = np.concatenate(indices) if indices else np.array([], dtype=int)
    records = data["records"].iloc[indices]
    result = {}
    for user in users:
        chosen = records.user_id.eq(user).to_numpy()
        counts = np.bincount(states[indices[chosen]] * 5 + records.loc[chosen, "action"].to_numpy(),
                             minlength=state_count * 5).reshape(state_count, 5)
        result[user] = (counts, activity.loc[activity.user_id.eq(user)])
    return result


def _fit_map_effects(rows, model, return_model, max_users):
    """Estimate source-supported offsets, with later observations tuning shrinkage."""
    if isinstance(max_users, bool) or not isinstance(max_users, int) or max_users < 2:
        raise ValueError("Gaussian fitting requires max_users >= 2")
    latent = model["latent_shopping"]
    available = canonical_available_actions(latent["encoder"].get("available_actions"))
    weights = np.asarray(latent["mixture_weights"])
    original = np.column_stack([latent["component_offsets"], latent["return_activity"]["component_logit_offsets"]])
    centre = weights @ original
    seed = int(model.get("seed", 42))
    # shortcut: hash-uniform capped users bound repeated Bellman fits; increase max_users for population precision.
    users = sorted(rows.user_id.astype(str).unique(),
        key=lambda user: hashlib.sha256(f"gaussian-rewards:{seed}:{user}".encode()).digest())[:max_users]
    selected = rows.loc[rows.user_id.astype(str).isin(users)].copy()
    memory = latent["encoder"].get("lifecycle_memory")
    kwargs = {"available_actions": available}
    if memory is not None:
        kwargs["lifecycle_memory"] = memory
    data = shopping.build_source(selected, model["observation_end"], seed, **kwargs)
    states, fallback = (shopping.encode(latent["encoder"], data["features"]) if memory is None else
                        shopping.encode(latent["encoder"], data["features"], phase_ids=source_phases.phase_ids(data["records"])))
    cutoff = pd.Timestamp(latent.get("source_cutoff_at", latent["return_activity"].get("cutoff_at", joint.CUTOFF)))
    masks = source_session_masks(data["sessions"], model["observation_end"], cutoff=cutoff)
    mask = np.asarray(latent["action_mask"], dtype=bool)
    all_indices = [np.arange(row.record_start, row.record_end)
                   for row in data["sessions"].loc[masks["all"]].itertuples()]
    all_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=int)
    if not mask[states[all_indices], data["records"].iloc[all_indices].action.to_numpy()].all():
        raise ValueError("source Gaussian fit contains an impossible fitted action")
    # Actual boundary users preserve the full-source activity clock after sampling.
    boundary_users = rows.loc[[rows["at"].idxmin(), rows["at"].idxmax()], "user_id"]
    activity_rows = rows.loc[rows.user_id.astype(str).isin([*users, *boundary_users.astype(str)])]
    activity, activity_audit = joint.build_activity_panel(activity_rows, return_model, seed)
    activity = activity.loc[activity.user_id.astype(str).isin(users)
        & activity.target_at.le(pd.Timestamp(model["observation_end"]))].copy()
    activity["user_id"] = activity.user_id.astype(str)
    activity_parts = {"all": activity, "prefix": activity.loc[activity.target_at.lt(cutoff)],
                      "later": activity.loc[activity["at"].ge(cutoff)]}
    observations = {name: _observations(data, states, chosen, activity_parts[name], users, len(mask))
                    for name, chosen in masks.items()}
    def exposure(observation):
        counts, visits = observation
        return int(counts.sum()) + len(visits)
    supported_users = [user for user in users if exposure(observations["all"][user])]
    tuning_users = [user for user in supported_users if exposure(observations["prefix"][user])
                    and exposure(observations["later"][user])]
    if len(supported_users) < 2 or not tuning_users:
        raise ValueError("Gaussian MAP requires supported users and strictly separated prefix/later observations")
    matrices = shopping.unpack_matrices(latent)
    base = np.asarray(latent["common_reward"])
    grid, convergence = [], []
    for penalty in PENALTIES:
        loss, count = 0., 0
        for user in tuning_users:
            theta, fit = fit_user_offsets(*observations["prefix"][user], base, matrices, mask, centre, penalty, available_actions=available)
            convergence.append(fit)
            later = observations["later"][user]
            loss += map_objective(theta, *later, base, matrices, mask, centre, 0., available_actions=available)[0]
            count += exposure(later)
        grid.append({"penalty": penalty, "later_negative_log_likelihood_per_observation": loss / count,
                     "users": len(tuning_users), "observations": count})
        print(json.dumps({"stage": "gaussian_map_tuning", **grid[-1]}), flush=True)
    penalty = min(grid, key=lambda item: item["later_negative_log_likelihood_per_observation"])["penalty"]
    coefficients = []
    for user in supported_users:
        theta, fit = fit_user_offsets(*observations["all"][user], base, matrices, mask, centre, penalty, available_actions=available)
        coefficients.append(theta);convergence.append(fit)
    report = {
        "max_users": max_users, "sampled_users": len(users), "supported_users": len(supported_users),
        "source_users": int(rows.user_id.nunique()), "selected_penalty": penalty, "tuning_grid": grid,
        "tuning_centre": "frozen population-weighted centroid; no full-source posterior used for tuning",
        "tuning_clock": "whole sessions terminal_known_at < cutoff vs start_at >= cutoff; activity target_at < cutoff vs at >= cutoff",
        "cutoff_at": cutoff.isoformat(), "action_observations": sum(int(observations["all"][u][0].sum()) for u in supported_users),
        "activity_observations": sum(len(observations["all"][u][1]) for u in supported_users),
        "map_fits": len(convergence), "all_map_fits_converged": True,
        "maximum_iterations": max(item["iterations"] for item in convergence),
        "maximum_gradient_abs": max(item["gradient_max_abs"] for item in convergence),
        "encoder_fallback_observations": int(fallback.sum()), "activity_panel": activity_audit,
    }
    return supported_users, np.asarray(coefficients), report


def fit_gaussian(rows, model, return_model, max_users=256):
    """Fit the legacy Gaussian moments using frozen discrete responsibilities."""
    started = time.perf_counter()
    supported_users, coefficients, report = _fit_map_effects(rows, model, return_model, max_users)
    latent = model["latent_shopping"]
    available = canonical_available_actions(latent["encoder"].get("available_actions"))
    weights = np.asarray(latent["mixture_weights"])
    original = np.column_stack([latent["component_offsets"], latent["return_activity"]["component_logit_offsets"]])
    responsibilities = [latent["profile_posteriors"].get(
        source_profile_id(model.get("source_contract", {}).get("domain", model.get("source", {}).get("domain", "cosmetics")), str(user)), latent["mixture_weights"])
        for user in supported_users]
    means, variances, masses = weighted_gaussian_moments(coefficients, responsibilities, original, available_actions=available)
    result = {"schema_version": 1, "type": TYPE, "purpose": "synthesis_only",
        "parameter_names": PARAMETERS, "component_count": len(weights), "mixture_weights": weights.tolist(),
        "means": means.tolist(), "variances": variances.tolist(),
        "fit": {"scope": FIT_SCOPE, "forecast_validation_uses_gaussian": False, **report,
            "effective_component_users": masses.tolist(), "unsupported_component_rule": "mass < 1e-6 preserves original mean and zero variance",
            "elapsed_seconds": time.perf_counter() - started},
        "limitations": ["Two-stage conditional Gaussian approximation with frozen discrete donor gating; not a jointly refitted continuous mixture.",
            "MAP shrinkage and the hash-uniform user sample affect dispersion; no invented covariance floor or extra jitter is added.",
            "Diagonal covariance does not preserve within-component action/return correlations.",
            "Full-source user coefficients and donor responsibilities are synthesis traits, never live/heldout forecasting predictors.",
            "The frozen parent's forecast scores do not validate the Gaussian model; tuning scores only select MAP shrinkage."]}
    if available != tuple(PARAMETERS[:4]):
        result.update(available_actions=list(available), fitted_parameter_names=[PARAMETERS[i] for i in parameter_indices(available)])
        result["fit"]["unobserved_parameter_rule"] = "MAP and Gaussian moments use identified coordinates only; other canonical slots are unused zero references, never observations."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "return-model", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--max-users", type=int, default=256)
    args = parser.parse_args()
    paths = (args.base, args.return_model, args.source_cache, args.output, args.report)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Gaussian source, models, sidecar and report paths must differ")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"immutable Gaussian output exists: {path}")
    model = json.loads(args.base.read_text())
    return_sha = _sha256(args.return_model)
    joint.validate_model(model, return_sha)
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    rows, descriptor = load_prior_source(model, args.source_cache)
    result = fit_gaussian(rows, model, json.loads(args.return_model.read_text()), args.max_users)
    result.update(base_model_sha256=_sha256(args.base), return_model_sha256=return_sha,
        provenance={"base_model_path": str(args.base), "return_model_path": str(args.return_model),
            "source_cache_path": str(args.source_cache), "source_cache_sha256": _sha256(args.source_cache),
            "source_descriptor": descriptor, "fitter_sha256": _sha256(__file__)})
    validate_gaussian(result, result["base_model_sha256"], return_sha)
    for path, payload in ((args.output, result), (args.report, {"passed": True, **result})):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"model_path": str(args.output), "users": result["fit"]["supported_users"],
        "selected_penalty": result["fit"]["selected_penalty"], "elapsed_seconds": result["fit"]["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
