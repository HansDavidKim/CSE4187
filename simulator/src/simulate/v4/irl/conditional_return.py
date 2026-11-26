"""Draw the fitted GMM return-logit offset given a discrete shopping offset."""

import numpy as np
from scipy.stats import chi2

from simulate.v4.irl.source_contract import canonical_available_actions


def prepare_conditional_return(block, discrete_action_offsets, discrete_return_offsets, *, available_actions=None):
    actions = canonical_available_actions(block.get("available_actions"))
    if available_actions is not None and actions != canonical_available_actions(available_actions):
        raise ValueError("conditional return action support differs from its caller")
    priors = np.asarray(block["mixture_weights"], dtype=float)
    means = np.asarray(block["means"], dtype=float)
    covariances = np.asarray(block["covariances"], dtype=float)
    action_offsets = np.asarray(discrete_action_offsets, dtype=float)
    fallback_offsets = np.asarray(discrete_return_offsets, dtype=float)
    size = len(priors)
    if (priors.shape != (size,) or not size or means.shape != (size, 5)
            or covariances.shape != (size, 5, 5) or action_offsets.ndim != 2
            or action_offsets.shape[1] != 4 or not len(action_offsets)
            or fallback_offsets.shape != (len(action_offsets),)):
        raise ValueError("invalid conditional return parameters")
    if actions != canonical_available_actions():
        observed = [canonical_available_actions().index(action) for action in actions]
        marginal = [*observed, 4]
        means = means[:, marginal]
        covariances = covariances[:, marginal][:, :, marginal]
        action_offsets = action_offsets[:, observed]
    if (not all(np.isfinite(a).all() for a in (priors, means, covariances, action_offsets, fallback_offsets))
            or (priors <= 0).any()):
        raise ValueError("invalid conditional return parameters")

    # The last diagonal of the joint Cholesky factor is the Schur standard deviation.
    factors = np.linalg.cholesky(covariances)
    action_factors = factors[:, :-1, :-1]
    centered = action_offsets[:, None, :] - means[None, :, :-1]
    whitened = np.linalg.solve(action_factors, centered.transpose(1, 2, 0)).transpose(2, 0, 1)
    return_means = means[None, :, -1] + np.einsum("nkd,kd->nk", whitened, factors[:, -1, :-1])
    return_scales = factors[:, -1, -1]

    log_determinant_half = np.log(np.diagonal(action_factors, axis1=1, axis2=2)).sum(axis=1)
    distance_squared = np.square(whitened).sum(axis=2)
    covered = distance_squared.min(axis=1) <= chi2.ppf(0.999, df=action_offsets.shape[1])
    log_weights = np.log(priors) - log_determinant_half - 0.5 * distance_squared
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)

    def sample(component, rng):
        if (isinstance(component, bool) or not isinstance(component, (int, np.integer))
                or not 0 <= component < len(action_offsets)):
            raise ValueError("invalid discrete shopping component")
        if not covered[component]:
            return float(fallback_offsets[component])
        index = int(rng.choice(size, p=weights[component]))
        return float(rng.normal(return_means[component, index], return_scales[index]))

    return sample
