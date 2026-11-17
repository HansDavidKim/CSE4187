"""Validate and sample the learned joint shopping and return reward mixture."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl.fit_gaussian_rewards import PARAMETERS


TYPES = ("source_joint_reward_gmm", "source_map_reward_gmm")


def validate_gmm(block, base_sha256, return_sha256):
    """Validate the sidecar and factor the component covariances once per run."""
    if (not isinstance(block, dict) or type(block.get("schema_version")) is not int
            or block["schema_version"] != 1 or block.get("type") not in TYPES
            or block.get("purpose") != "synthesis_only" or block.get("parameter_names") != PARAMETERS):
        raise ValueError("incompatible reward GMM sidecar")
    for key, expected in (("base_model_sha256", base_sha256), ("return_model_sha256", return_sha256)):
        if block.get(key) != expected:
            raise ValueError(f"reward GMM requires its exact frozen {key}")
    count = block.get("component_count")
    if type(count) is not int or count < 2:
        raise ValueError("reward GMM requires at least two components")
    means = np.asarray(block.get("means"), dtype=float)
    covariances = np.asarray(block.get("covariances"), dtype=float)
    weights = np.asarray(block.get("mixture_weights"), dtype=float)
    width = len(PARAMETERS)
    if (means.shape != (count, width) or covariances.shape != (count, width, width)
            or weights.shape != (count,) or not all(np.isfinite(a).all() for a in (means, covariances, weights))
            or (weights <= 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10)
            or not np.allclose(covariances, covariances.swapaxes(1, 2), rtol=1e-10, atol=1e-12)):
        raise ValueError("invalid reward GMM parameters")
    posteriors = block.get("profile_posteriors", {})
    if not isinstance(posteriors, dict):
        raise ValueError("reward GMM donor posteriors must be keyed by profile")
    for posterior in posteriors.values():
        values = np.asarray(posterior, dtype=float)
        if (values.shape != weights.shape or not np.isfinite(values).all()
                or (values < 0).any() or not np.isclose(values.sum(), 1, rtol=0, atol=1e-10)):
            raise ValueError("invalid reward GMM donor posterior")
    try:
        return np.linalg.cholesky(covariances)
    except np.linalg.LinAlgError as exc:
        raise ValueError("reward GMM covariances must be positive definite") from exc


def prepare_bellman(latent, matrices):
    """Pack the fixed Bellman transitions once for all customer draws."""
    base = np.asarray(latent["common_reward"])
    mask = np.asarray(latent["action_mask"], dtype=bool)
    anchors = np.asarray(latent["component_offsets"])
    values = np.asarray(latent["component_values"])
    states = len(base)
    packed = (shopping.DISCOUNT * sparse.vstack(matrices, format="csr")).tocoo()
    transition = packed.tocsr()
    diagonal = np.arange(states) * (states + 1)
    coordinates, slots = np.unique(
        np.r_[packed.row % states * states + packed.col, diagonal], return_inverse=True)
    jacobian = sparse.csr_matrix(
        (np.ones(len(coordinates)), (coordinates // states, coordinates % states)), shape=(states, states))
    ones = np.ones(states)

    def policy(offsets):
        reward = shopping._reward(base + offsets)
        value = values[np.argmin(np.square(anchors - offsets).sum(axis=1))]
        for _ in range(16):
            q = np.where(mask, reward + (transition @ value).reshape(len(matrices), states).T, -np.inf)
            maximum = q.max(axis=1)
            mass = np.exp(q - maximum[:, None])
            total = mass.sum(axis=1)
            residual = maximum + np.log(total) - value
            if not np.isfinite(residual).all():
                break
            probabilities = mass / total[:, None]
            if np.max(np.abs(residual)) < 1e-10:
                return probabilities
            # Newton solves (I - gamma P_pi) delta = Bellman(value) - value.
            coefficients = -packed.data * probabilities.T.ravel()[packed.row]
            data = np.bincount(slots, weights=np.r_[coefficients, ones], minlength=len(coordinates))
            system = sparse.csr_matrix((data, jacobian.indices, jacobian.indptr), shape=jacobian.shape)
            try:
                step = spsolve(system, residual)
            except (RuntimeError, np.linalg.LinAlgError):
                break
            if not np.isfinite(step).all():
                break
            value = value + step
        return shopping.soft_policy(reward, matrices, mask)[0]

    return policy


def sample_reward(block, factors, solve_policy, profile_id, rng):
    posterior = block.get("profile_posteriors", {}).get(profile_id)
    weights = block["mixture_weights"] if posterior is None else posterior
    scope = "gmm_population_prior" if posterior is None else "gmm_source_donor_posterior"
    component = int(rng.choice(len(weights), p=weights))
    theta = np.asarray(block["means"][component]) + factors[component] @ rng.standard_normal(len(PARAMETERS))
    policy = solve_policy(theta[:4])
    return component, scope, {"action_offsets": theta[:4], "return_offset": float(theta[4]), "policy": policy}
