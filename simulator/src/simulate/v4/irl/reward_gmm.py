"""Validate and sample the learned joint shopping and return reward mixture."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl.fit_gaussian_rewards import PARAMETERS, parameter_indices, validate_parameter_support
from simulate.v4.irl.source_contract import canonical_available_actions


TYPES = ("source_joint_reward_gmm", "source_map_reward_gmm", "source_persona_joint_reward_gmm")


def validate_gmm(block, base_sha256, return_sha256, *, available_actions=None):
    """Validate the sidecar and factor the component covariances once per run."""
    if (not isinstance(block, dict) or type(block.get("schema_version")) is not int
            or block["schema_version"] != 1 or block.get("type") not in TYPES
            or block.get("purpose") != "synthesis_only" or block.get("parameter_names") != PARAMETERS):
        raise ValueError("incompatible reward GMM sidecar")
    if block.get("backend", "retail_session") != "retail_session":
        raise ValueError("daily reward coordinates cannot enter the retail return-logit backend")
    for key, expected in (("base_model_sha256", base_sha256), ("return_model_sha256", return_sha256)):
        if block.get(key) != expected:
            raise ValueError(f"reward GMM requires its exact frozen {key}")
    count = block.get("component_count")
    if type(count) is not int or count < 2:
        raise ValueError("reward GMM requires at least two components")
    if "component_model_indices" in block:
        indices = block["component_model_indices"]
        if (not isinstance(indices, list) or len(indices) != count
                or any(type(index) is not int or index < 0 for index in indices)):
            raise ValueError("reward GMM needs one valid actor index per component")
    if block["type"] == "source_persona_joint_reward_gmm" and "component_model_indices" not in block:
        raise ValueError("persona GMM must bind its frozen persona actors")
    if block["type"] == "source_persona_joint_reward_gmm":
        labels = block.get("persona_labels")
        if (not isinstance(labels, list) or len(labels) != count
                or any(not isinstance(label, str) or not label for label in labels)
                or len(set(labels)) != count or len(set(block["component_model_indices"])) != count):
            raise ValueError("persona GMM needs distinct labels and one-to-one actor bindings")
    conditional = block.get("condition_on_profile_components", False)
    if type(conditional) is not bool or conditional and sorted(block.get("component_model_indices", [])) != list(range(count)):
        raise ValueError("profile conditioning requires a one-to-one persona/actor mapping")
    for name, zero_allowed in (("choice_temperatures", False), ("dispersion_scales", True)):
        if name in block:
            values = np.asarray(block[name], dtype=float)
            if (values.shape != (count,) or not np.isfinite(values).all()
                    or any(isinstance(value, (bool, np.bool_)) for value in block[name])
                    or (values < 0).any() or not zero_allowed and (values == 0).any()):
                raise ValueError("invalid reward GMM temperature or dispersion scale")
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
    available = validate_parameter_support(block, available_actions)
    active = parameter_indices(available)
    try:
        return np.linalg.cholesky(covariances if len(active) == 5 else covariances[:, active][:, :, active])
    except np.linalg.LinAlgError as exc:
        raise ValueError("reward GMM covariances must be positive definite") from exc


def prepare_bellman(latent, matrices):
    """Pack the fixed Bellman transitions once for all customer draws."""
    base = np.asarray(latent["common_reward"])
    mask = np.asarray(latent["action_mask"], dtype=bool)
    anchors = np.asarray(latent["component_offsets"])
    available = canonical_available_actions(latent.get("encoder", {}).get("available_actions"))
    supported = [PARAMETERS.index(a) for a in available]
    values = np.asarray(latent["component_values"])
    deviations = (np.asarray(latent["component_reward_deviations"])
                  if "component_reward_deviations" in latent else None)
    if deviations is not None and (deviations.shape != (len(anchors), *base.shape) or not np.isfinite(deviations).all()):
        raise ValueError("invalid persona reward deviations")
    states = len(base)
    packed = (shopping.DISCOUNT * sparse.vstack(matrices, format="csr")).tocoo()
    transition = packed.tocsr()
    diagonal = np.arange(states) * (states + 1)
    coordinates, slots = np.unique(
        np.r_[packed.row % states * states + packed.col, diagonal], return_inverse=True)
    jacobian = sparse.csr_matrix(
        (np.ones(len(coordinates)), (coordinates // states, coordinates % states)), shape=(states, states))
    ones = np.ones(states)

    def policy(offsets, *, component_id=None, temperature=1.):
        if (isinstance(temperature, (bool, np.bool_)) or not np.isfinite(temperature) or temperature <= 0):
            raise ValueError("choice temperature must be finite and positive")
        if component_id is not None and (isinstance(component_id, (bool, np.bool_))
                or not isinstance(component_id, (int, np.integer)) or not 0 <= component_id < len(anchors)):
            raise ValueError("invalid frozen persona actor index")
        if deviations is not None and component_id is None:
            raise ValueError("persona reward deviations require an explicit actor index")
        if len(supported) != 4:
            offsets = np.asarray(offsets).copy()
            offsets[[i for i in range(4) if i not in supported]] = 0.
        actor_base = base if deviations is None else base + deviations[int(component_id)]
        reward = shopping._reward(actor_base + offsets)
        if temperature != 1.:
            reward = reward / temperature
        anchor = np.argmin(np.square(anchors - offsets).sum(axis=1)) if component_id is None else int(component_id)
        value = values[anchor] if temperature == 1. else values[anchor] / temperature
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


def sample_reward(block, factors, solve_policy, profile_id, rng, *, profile_components=None):
    if block.get("type") == "source_persona_joint_choice_gmm":
        raise ValueError("choice-logit tastes require the separate choice GMM sampler")
    if block.get("backend", "retail_session") != "retail_session":
        raise ValueError("daily reward coordinates require their daily policy sampler")
    posterior = block.get("profile_posteriors", {}).get(profile_id)
    weights = block["mixture_weights"] if posterior is None else posterior
    scope = "gmm_population_prior" if posterior is None else "gmm_source_donor_posterior"
    if block.get("condition_on_profile_components", False):
        values = np.asarray(profile_components, dtype=float)
        indices = block.get("component_model_indices", [])
        if (values.shape != (len(weights),) or not np.isfinite(values).all() or (values < 0).any()
                or not np.isclose(values.sum(), 1., atol=1e-10, rtol=0)
                or sorted(indices) != list(range(len(weights)))):
            raise ValueError("profile persona probabilities require a matching one-to-one actor mapping")
        weights = values[indices]
        scope = "gmm_profile_persona_prior"
    elif profile_components is not None:
        raise ValueError("anonymous GMM cannot condition on actor profile probabilities")
    component = int(rng.choice(len(weights), p=weights))
    active = parameter_indices(block.get("available_actions"))
    if len(active) == 5:
        noise = factors[component] @ rng.standard_normal(len(PARAMETERS))
        if "dispersion_scales" in block:
            noise = noise * block["dispersion_scales"][component]
        theta = np.asarray(block["means"][component]) + noise
    else:
        theta = np.zeros(5)
        noise = factors[component] @ rng.standard_normal(len(active))
        if "dispersion_scales" in block:
            noise = noise * block["dispersion_scales"][component]
        theta[active] = np.asarray(block["means"][component])[active] + noise
    options = {}
    if "component_model_indices" in block:
        options["component_id"] = block["component_model_indices"][component]
    if "choice_temperatures" in block:
        options["temperature"] = block["choice_temperatures"][component]
    policy = solve_policy(theta[:4], **options)
    return component, scope, {"action_offsets": theta[:4], "return_offset": float(theta[4]), "policy": policy}
