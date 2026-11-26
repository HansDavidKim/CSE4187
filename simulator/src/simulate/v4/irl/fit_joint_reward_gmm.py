"""Fit a reward GMM directly to observed shopping decisions and return windows."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from simulate.v4.irl import fit_joint_latent_shopping as joint
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl import source_phases
from simulate.v4.irl.fit_donor_prior import load_prior_source
from simulate.v4.irl.fit_gaussian_rewards import PARAMETERS, _sha256, source_session_masks, parameter_indices
from simulate.v4.irl.source_contract import canonical_available_actions, source_profile_id
from simulate.v4.irl.reward_gmm import validate_gmm


TYPE = "source_joint_reward_gmm"
COMPONENTS = 3
MAX_UPDATES = 8
EIGENVALUE_FLOOR = 0.0004
TRAIN_SCRAMBLE = 293
TRAIN_POWER = 7
CHECK_SCRAMBLES = (751, 1293)
INITIAL_SD = (.11, .07, .07, .06, .3)


def sobol_normal(power, seed, *, dimension=len(PARAMETERS)):
    return ndtri(qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(power))


def draws(means, covariances, normal, *, available_actions=None):
    active = parameter_indices(available_actions)
    if len(active) == 5:
        return means[:, None, :] + normal[None, :, :] @ np.linalg.cholesky(covariances).swapaxes(1, 2)
    result = np.zeros((len(means), len(normal), 5))
    marginal = np.asarray(covariances)[:, active][:, :, active]
    result[:, :, active] = np.asarray(means)[:, None, active] + normal[None, :, :] @ np.linalg.cholesky(marginal).swapaxes(1, 2)
    return result


def policy_logs(base, mask, transition, offsets):
    """Return finite log policy columns for sparse state-action count sums."""
    blocks = []
    for start in range(0, len(offsets), 256):
        coefficients = offsets[start:start+256, :4]
        reward = shopping._reward(base)[:, :, None] + shopping._reward(coefficients).T[None, :, :]
        reward = np.where(mask[:, :, None], reward, -np.inf).transpose(1, 0, 2)
        values = np.zeros((len(base), len(coefficients)))
        for _ in range(5000):
            q = reward + (transition @ values).reshape(5, len(base), -1)
            updated = logsumexp(q, axis=0)
            difference = np.max(np.abs(updated - values))
            values = updated
            if difference < 1e-9:
                break
        else:
            raise ValueError("batched soft Bellman iteration did not converge")
        q = reward + (transition @ values).reshape(5, len(base), -1)
        logp = q - logsumexp(q, axis=0, keepdims=True)
        # Unsupported actions are validated against the mask before computing likelihood.
        logp = np.where(mask.T[:, :, None], logp, 0.)
        blocks.append(logp.transpose(2, 1, 0).reshape(len(coefficients), -1))
    return np.concatenate(blocks)


def observations(data, states, mask, session_mask, activity):
    indices = shopping.valid_record_indices(data, session_mask)
    records = data["records"].iloc[indices]
    if not mask[states[indices], records.action.to_numpy()].all():
        raise ValueError("source GMM received an unsupported action after encoding")
    users, counts, incidence, _ = joint.joint_counts({"records": records}, states[indices], len(mask), activity)
    if not len(users) or not len(records) or not len(activity):
        raise ValueError("reward GMM requires both action and activity observations")
    return users, counts, incidence, activity


def likelihood_terms(panel, offsets, log_policy):
    _, counts, incidence, activity = panel
    return counts @ log_policy.T + incidence @ joint.activity_log_probability(activity, offsets)


def joint_loss(panel, sample, weights, log_policy, moments=False):
    k, count, width = sample.shape
    logits = likelihood_terms(panel, sample.reshape(-1, width)[:, -1], log_policy)
    logits += np.log(weights).repeat(count)[None, :] - np.log(count)
    evidence = logsumexp(logits, axis=1)
    nll = float(-evidence.sum() / (panel[1].sum() + len(panel[3])))
    if not moments:
        return nll
    responsibility = np.exp(logits - evidence[:, None]).sum(axis=0).reshape(k, count)
    return nll, responsibility


def update_moments(sample, responsibility, *, available_actions=None):
    mass = responsibility.sum(axis=1)
    if not np.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError("a fitted reward GMM component has no posterior mass")
    active = parameter_indices(available_actions)
    values = sample if len(active) == 5 else sample[:, :, active]
    means = np.einsum("ki,kij->kj", responsibility, values) / mass[:, None]
    centered = values - means[:, None, :]
    covariance = np.einsum("ki,kij,kil->kjl", responsibility, centered, centered) / mass[:, None, None]
    eigenvalues, vectors = np.linalg.eigh(covariance)
    floored = (vectors * np.maximum(eigenvalues[:, None, :], EIGENVALUE_FLOOR)) @ vectors.swapaxes(1, 2)
    if len(active) == 5:
        return mass / mass.sum(), means, floored
    embedded_mean = np.zeros((len(mass), 5))
    embedded_mean[:, active] = means
    embedded_covariance = np.repeat(np.diag(np.square(INITIAL_SD))[None, :, :], len(mass), axis=0)
    embedded_covariance[:, np.asarray(active)[:, None], active] = floored
    return mass / mass.sum(), embedded_mean, embedded_covariance


def donor_posteriors(model, panel, parameters, base, mask, transition):
    weights, means, covariance = parameters
    available = canonical_available_actions(model["latent_shopping"]["encoder"].get("available_actions"))
    seed = 20261117
    sample = draws(means, covariance, sobol_normal(TRAIN_POWER + 1, seed, dimension=len(parameter_indices(available))), available_actions=available)
    count = sample.shape[1]
    flat = sample.reshape(-1, len(PARAMETERS))
    log_policy = policy_logs(base, mask, transition, flat)
    users, counts, incidence, activity = panel
    activity_logs = joint.activity_log_probability(activity, flat[:, -1])
    profiles = {profile["profile_id"] for profile in model["profiles"]}
    result = {}
    for start in range(0, len(users), 512):
        stop = min(start + 512, len(users))
        logs = counts[start:stop] @ log_policy.T + incidence[start:stop] @ activity_logs
        logs = logsumexp(logs.reshape(stop - start, len(weights), count), axis=2)
        logs += np.log(weights)[None, :] - np.log(count)
        probabilities = np.exp(logs - logsumexp(logs, axis=1, keepdims=True))
        for user, probability in zip(users[start:stop], probabilities, strict=True):
            profile_id = source_profile_id(model.get("source_contract", {}).get("domain", model.get("source", {}).get("domain", "cosmetics")), str(user))
            if profile_id in profiles:
                result[profile_id] = probability.tolist()
    if not result:
        raise ValueError("no supported histories matched frozen donor profile identifiers")
    return result, {
        "purpose": "synthesis_only",
        "rule": "posterior component weights conditioned on recorded donor history; draw within-component offsets from the fitted prior",
        "known_donors": len(result), "source_profiles": len(profiles),
        "activity_windows": len(activity), "action_decisions": int(counts.sum()),
        "scramble": seed, "points_per_component": count,
    }


def prepared_calibration_masks(sessions, activity, model, cutoff):
    """Prepared recipes reserve outer users and all post-fit labels from selection."""
    contract = model["source_contract"]
    declared = contract.get("training_cutoff_at", contract.get("outer_cutoff_at"))
    if declared is None or pd.Timestamp(declared) != cutoff:
        raise ValueError("GMM calibration cutoff differs from its prepared source contract")
    seed = int(model.get("seed", 42))
    users = set(sessions.user_id) | set(activity.user_id)
    outer = {uid for uid in users if sha256(f"{seed}:{uid}".encode()).digest()[0] < 205}
    train = {uid for uid in outer if sha256(f"{seed}:inner:{uid}".encode()).digest()[0] < 205}
    selection = outer - train
    if "train_user" in activity and not activity.train_user.eq(activity.user_id.isin(outer)).all():
        raise ValueError("GMM activity users differ from the frozen source outer split")
    prefix = source_session_masks(sessions, model["observation_end"], cutoff=cutoff)["prefix"]
    mature = activity.target_at.lt(cutoff)
    masks = {"train_sessions": prefix & sessions.user_id.isin(train),
             "selection_sessions": prefix & sessions.user_id.isin(selection),
             "train_activity": mature & activity.user_id.isin(train),
             "selection_activity": mature & activity.user_id.isin(selection),
             "donor_sessions": prefix & sessions.user_id.isin(outer),
             "donor_activity": mature & activity.user_id.isin(outer)}
    audit = {"outer_holdout_users_excluded": len(users - outer),
             "optimization_users": len(train), "internal_selection_users": len(selection),
             "sessions_without_valid_pre_cutoff_support_excluded": int((~prefix).sum()),
             "activity_not_mature_before_fit_cutoff_excluded": int((~mature).sum()),
             "outer_holdout_and_postcutoff_labels_used_for_selection": False,
             "scope": "conditional GMM calibration: frozen base actor/return heads used the full outer training partition, including these internal selection users"}
    for name in ("train", "selection"):
        ends = sessions.loc[masks[f"{name}_sessions"], "terminal_known_at"]
        targets = activity.loc[masks[f"{name}_activity"], "target_at"]
        audit[f"max_{name}_action_known_at"] = ends.max().isoformat() if len(ends) else None
        audit[f"max_{name}_activity_known_at"] = targets.max().isoformat() if len(targets) else None
    return masks, audit


def fit_joint_reward_gmm(rows, model, return_model, *, with_donor_posteriors=False):
    latent = model["latent_shopping"]
    available = canonical_available_actions(latent["encoder"].get("available_actions"))
    base = np.asarray(latent["common_reward"], dtype=float)
    mask = np.asarray(latent["action_mask"], dtype=bool)
    matrices = shopping.unpack_matrices(latent)
    transition = shopping.DISCOUNT * sparse.vstack(matrices, format="csr")
    memory = latent["encoder"].get("lifecycle_memory")
    kwargs = {"available_actions": available}
    if memory is not None:
        kwargs["lifecycle_memory"] = memory
    source = shopping.build_source(rows, model["observation_end"], model["seed"], **kwargs)
    states, _ = (shopping.encode(latent["encoder"], source["features"]) if memory is None else
                 shopping.encode(latent["encoder"], source["features"], phase_ids=source_phases.phase_ids(source["records"])))
    sessions = source["sessions"]
    cutoff = pd.Timestamp(latent.get("source_cutoff_at", latent["return_activity"].get("cutoff_at", joint.CUTOFF)))
    masks = source_session_masks(sessions, model["observation_end"], cutoff=cutoff)
    activity, _ = joint.build_activity_panel(rows, return_model, model["seed"])
    activity = activity.loc[activity.target_at.le(pd.Timestamp(model["observation_end"]))]
    calibration_masks = calibration_audit = None
    if "source_contract" in model:
        calibration_masks, calibration_audit = prepared_calibration_masks(sessions, activity, model, cutoff)
        train = observations(source, states, mask, calibration_masks["train_sessions"],
                             activity.loc[calibration_masks["train_activity"]])
        validation = observations(source, states, mask, calibration_masks["selection_sessions"],
                                  activity.loc[calibration_masks["selection_activity"]])
    else:
        train_users = {
            user for user in rows.user_id.unique()
            if sha256(f"reward-gmm:user-split:{user}".encode()).digest()[0] < 205
        }
        train = observations(source, states, mask, masks["prefix"] & sessions.user_id.isin(train_users),
            activity.loc[activity.target_at.lt(cutoff) & activity.user_id.isin(train_users)])
        validation = observations(source, states, mask, masks["later"] & ~sessions.user_id.isin(train_users),
            activity.loc[activity["at"].ge(cutoff) & ~activity.user_id.isin(train_users)])
    if np.intersect1d(train[0], validation[0]).size:
        raise ValueError("reward GMM train and validation users overlap")

    discrete_weights = np.asarray(latent["mixture_weights"])
    discrete_offsets = np.c_[latent["component_offsets"], latent["return_activity"]["component_logit_offsets"]]
    baseline = {name: -joint.joint_marginal(base, discrete_offsets[:, :4], discrete_offsets[:, 4],
        discrete_weights, panel[1], panel[2], panel[3], matrices, mask, 0)[1]
        for name, panel in (("train", train), ("internal_validation", validation))}
    selected = np.argsort(discrete_weights)[-COMPONENTS:]
    initial_weights = discrete_weights[selected] / discrete_weights[selected].sum()
    initial_covariance = np.repeat(np.diag(np.square(INITIAL_SD))[None, :, :], COMPONENTS, axis=0)
    parameters = initial_weights, discrete_offsets[selected], initial_covariance
    normal = sobol_normal(TRAIN_POWER, TRAIN_SCRAMBLE, dimension=len(parameter_indices(available)))
    trace = []
    best_nll = np.inf
    for step in range(MAX_UPDATES + 1):
        weights, means, covariance = parameters
        sample = draws(means, covariance, normal, available_actions=available)
        log_policy = policy_logs(base, mask, transition, sample.reshape(-1, len(PARAMETERS)))
        train_nll, responsibilities = joint_loss(train, sample, weights, log_policy, moments=True)
        validation_nll = joint_loss(validation, sample, weights, log_policy)
        effective_points = np.square(responsibilities.sum(axis=1)) / np.square(responsibilities).sum(axis=1)
        trace.append({"iteration": step, "nll_train": train_nll, "nll_internal_validation": validation_nll,
            "weights": weights.tolist(), "component_aggregate_ess": effective_points.tolist()})
        if validation_nll < best_nll:
            best_iteration, best_nll, best_parameters = step, validation_nll, parameters
        if step < MAX_UPDATES:
            parameters = update_moments(sample, responsibilities, available_actions=available)
    weights, means, covariance = best_parameters
    checks = []
    for seed in CHECK_SCRAMBLES:
        sample = draws(means, covariance, sobol_normal(TRAIN_POWER + 1, seed, dimension=len(parameter_indices(available))), available_actions=available)
        log_policy = policy_logs(base, mask, transition, sample.reshape(-1, len(PARAMETERS)))
        checks.append({"seed": seed, "nll_train": joint_loss(train, sample, weights, log_policy),
            "nll_internal_validation": joint_loss(validation, sample, weights, log_policy)})
    result = {
        "schema_version": 1, "type": TYPE, "purpose": "synthesis_only", "parameter_names": PARAMETERS,
        "component_count": COMPONENTS, "mixture_weights": weights.tolist(), "means": means.tolist(),
        "covariances": covariance.tolist(),
        "fit": {
            "scope": "training_customers_supported_sessions_and_mature_activity_before_cutoff",
            "algorithm": "approximate EM on the joint frozen-policy action and 30-day activity likelihood",
            "quadrature": "component-adaptive scrambled Sobol transformed by full Gaussian covariance",
            "covariance_eigenvalue_floor": EIGENVALUE_FLOOR, "initial_covariance_sd": list(INITIAL_SD),
            "candidate_selection": "minimum joint observation NLL on the distinct-user post-cutoff internal panel",
            "initialization": "three highest-mass components of the frozen discrete mixture",
            "train_scramble": TRAIN_SCRAMBLE, "points_per_component": 2**TRAIN_POWER,
            "split": "SHA256 first byte of reward-gmm:user-split:<user> below 205; fit pre-cutoff; check other users post-cutoff",
            "cutoff_at": cutoff.isoformat(), "stop_reason": "iteration_budget", "performed_updates": MAX_UPDATES,
            "selected_iteration": best_iteration,
            "discrete_v10_nll": baseline, "independent_quadrature_checks": checks, "trace": trace,
            "panels": {name: {"users": len(panel[0]), "action_decisions": int(panel[1].sum()),
                "activity_windows": len(panel[3])} for name, panel in (("train", train), ("internal_validation", validation))},
        },
    }
    if calibration_audit is not None:
        result["fit"].update(candidate_selection="minimum conditional GMM joint NLL on source-outer-train inner-holdout users with all labels strictly before fit cutoff",
            split="sha256(seed:user_id) outer train only; optimize inner train; select inner user holdout; both mature before fit cutoff",
            calibration_split_audit=calibration_audit,
            external_future_diagnostics_used_for_selection=False)
    if available != tuple(PARAMETERS[:4]):
        result.update(available_actions=list(available), fitted_parameter_names=[PARAMETERS[i] for i in parameter_indices(available)])
        result["fit"]["unobserved_parameter_rule"] = "Exact Gaussian marginal over identified coordinates; unused canonical slots have zero mean and independent fixed reference variance, not fitted evidence."
    if with_donor_posteriors:
        donor = observations(source, states, mask,
            calibration_masks["donor_sessions"] if calibration_masks is not None else masks["all"],
            activity.loc[calibration_masks["donor_activity"]] if calibration_masks is not None else activity)
        result["profile_posteriors"], result["donor_posterior_fit"] = donor_posteriors(
            model, donor, best_parameters, base, mask, transition)
        if calibration_audit is not None:
            result["donor_posterior_fit"]["availability_rule"] = "outer training users only; valid END and activity outcomes strictly before the fitted cutoff"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "return-model", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--with-donor-posteriors", action="store_true", help="add synthesis-only donor component probabilities")
    args = parser.parse_args()
    paths = (args.base, args.return_model, args.source_cache, args.output, args.report)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("GMM source, models, sidecar and report paths must differ")
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"immutable GMM output exists: {path}")
    started = time.monotonic()
    base = args.base.read_bytes()
    returning = args.return_model.read_bytes()
    base_sha, return_sha = (sha256(blob).hexdigest() for blob in (base, returning))
    model = json.loads(base)
    joint.validate_model(model, return_sha)
    rows, descriptor = load_prior_source(model, args.source_cache)
    sidecar = fit_joint_reward_gmm(rows, model, json.loads(returning), with_donor_posteriors=args.with_donor_posteriors)
    sidecar.update(base_model_sha256=base_sha, return_model_sha256=return_sha,
        provenance={"source_cache_sha256": _sha256(args.source_cache),
            "source_fingerprint": descriptor["fingerprint"], "fitter_sha256": _sha256(__file__)})
    validate_gmm(sidecar, base_sha, return_sha)
    report = {key: value for key, value in sidecar.items() if key != "profile_posteriors"}
    report.update(passed=True, elapsed_seconds=time.monotonic() - started, source_descriptor=descriptor)
    outputs = [(path, json.dumps(payload, indent=2, allow_nan=False))
               for path, payload in ((args.output, sidecar), (args.report, report))]
    created = []
    try:
        for path, text in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(text)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    print(json.dumps({"model_path": str(args.output), "components": COMPONENTS,
        "selected_iteration": sidecar["fit"]["selected_iteration"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
