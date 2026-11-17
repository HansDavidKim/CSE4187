"""Continuous rewards retain Bellman dynamics, clock boundaries and source bindings."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import fit_gaussian_rewards as gaussian
from simulate.v4.irl import fit_latent_shopping as shopping
from test_v4_latent_shopping import mdp


def sidecar():
    return dict(schema_version=1, type=gaussian.TYPE, purpose="synthesis_only",
        parameter_names=gaussian.PARAMETERS.copy(), component_count=2,
        mixture_weights=[.4, .6], means=[[0.] * 5, [.2] * 5], variances=[[0.] * 5] * 2,
        base_model_sha256="a" * 64, return_model_sha256="b" * 64,
        fit=dict(scope=gaussian.FIT_SCOPE, forecast_validation_uses_gaussian=False))


def source_model():
    matrices, mask = mdp()
    base = np.arange(12).reshape(3, 4) * .02
    offsets = np.array([[0., .1, -.2, .4], [.2, -.3, .1, -.4]])
    policies = [shopping.soft_policy(shopping._reward(base + row), matrices, mask)[0]
                for row in offsets]
    return dict(common_reward=base.tolist(), component_offsets=offsets.tolist(),
        component_policies=np.asarray(policies).tolist(), mixture_weights=[.4, .6],
        action_mask=mask.tolist()), matrices


def test_gaussian_map_exact_bellman_and_activity_gradient():
    matrices, mask = mdp()
    base = np.arange(12).reshape(3, 4) * .02
    counts = np.zeros((3, 5))
    counts[0, :2] = [3, 2];counts[1, [3, 4]] = [2, 1];counts[2, 4] = 4
    activity = pd.DataFrame(dict(base_logit=[-.3, .8, .1], label=[0, 1, 1]))
    theta = np.array([.2, -.1, .4, -.3, .5]);centre = np.zeros(5)
    _, derivative = gaussian.map_objective(theta, counts, activity, base, matrices, mask, centre, .5)
    for index in range(5):
        delta = np.zeros(5);delta[index] = 1e-5
        upper = gaussian.map_objective(theta + delta, counts, activity, base, matrices, mask, centre, .5)[0]
        lower = gaussian.map_objective(theta - delta, counts, activity, base, matrices, mask, centre, .5)[0]
        assert derivative[index] == pytest.approx((upper - lower) / 2e-5, abs=1e-6)
    fitted, evidence = gaussian.fit_user_offsets(counts, activity, base, matrices, mask, centre, .5)
    assert np.isfinite(fitted).all() and evidence["gradient_max_abs"] < 1e-5


def test_zero_variance_centroid_recovers_policy_and_preserves_rng():
    model, matrices = source_model();block = sidecar()
    block["means"] = [np.r_[row, return_value].tolist()
        for row, return_value in zip(model["component_offsets"], [-.2, .3])]
    rng = np.random.default_rng(17);reference = np.random.default_rng(17)
    draw = gaussian.sample_reward_policy(model, block, 1, rng, matrices)
    np.testing.assert_array_equal(draw["policy"], model["component_policies"][1])
    np.testing.assert_array_equal(draw["action_offsets"], model["component_offsets"][1])
    assert draw["return_offset"] == .3
    assert rng.random() == reference.random()


def test_positive_variance_draws_are_seeded_and_vary_inside_one_component():
    model, matrices = source_model();block = sidecar()
    block["variances"] = [[.04] * 5, [.09] * 5]
    first = gaussian.sample_reward_policy(model, block, 0, np.random.default_rng(8), matrices)
    same = gaussian.sample_reward_policy(model, block, 0, np.random.default_rng(8), matrices)
    other = gaussian.sample_reward_policy(model, block, 0, np.random.default_rng(9), matrices)
    for key in ("action_offsets", "policy"):
        np.testing.assert_array_equal(first[key], same[key])
        assert not np.array_equal(first[key], other[key])
    assert first["return_offset"] == same["return_offset"]
    np.testing.assert_allclose(first["policy"].sum(axis=1), 1)
    assert not first["policy"][~np.asarray(model["action_mask"])].any()
    original = np.asarray(model["component_policies"])
    gaussian.sample_reward_policy(model, block, 0, np.random.default_rng(10), matrices)
    np.testing.assert_array_equal(model["component_policies"], original)


@pytest.mark.parametrize("key,value", [
    ("schema_version", True), ("type", "noise"), ("purpose", "forecast"),
    ("base_model_sha256", "c" * 64), ("return_model_sha256", "c" * 64),
    ("component_count", True), ("component_count", 3),
    ("means", [[0.] * 4] * 2), ("means", [[float("inf")] * 5] * 2),
    ("variances", [[-.1] * 5] * 2), ("variances", [[float("nan")] * 5] * 2),
    ("mixture_weights", [.2, .2]), ("parameter_names", ["wrong"] * 5),
    ("fit", dict(scope="future", forecast_validation_uses_gaussian=False)),
    ("fit", dict(scope=gaussian.FIT_SCOPE, forecast_validation_uses_gaussian=True)),
])
def test_gaussian_sidecar_rejects_invalid_shape_binding_and_fit_scope(key, value):
    block = sidecar();gaussian.validate_gaussian(block, "a" * 64, "b" * 64)
    invalid = deepcopy(block);invalid[key] = value
    with pytest.raises(ValueError):
        gaussian.validate_gaussian(invalid, "a" * 64, "b" * 64)


@pytest.mark.parametrize("component", [True, -1, 2, 1.5])
def test_gaussian_sampler_rejects_invalid_components(component):
    model, matrices = source_model()
    with pytest.raises(ValueError):
        gaussian.sample_reward_policy(model, sidecar(), component, np.random.default_rng(1), matrices)


def test_weighted_source_map_moments_preserve_unidentified_component_without_jitter():
    coefficients = np.array([[0.] * 5, [2.] * 5, [4.] * 5])
    responsibilities = np.array([[1., 0., 0.], [.5, .5, 0.], [0., 1., 0.]])
    original = np.array([[5.] * 5, [6.] * 5, [7.] * 5])
    means, variances, mass = gaussian.weighted_gaussian_moments(coefficients, responsibilities, original)
    np.testing.assert_allclose(means[0], 2 / 3)
    np.testing.assert_allclose(means[1], 10 / 3)
    np.testing.assert_allclose(variances[:2], 8 / 9)
    np.testing.assert_array_equal(means[2], original[2])
    np.testing.assert_array_equal(variances[2], np.zeros(5))
    np.testing.assert_allclose(mass, [1.5, 1.5, 0.])


def test_tuning_clock_keeps_whole_sessions_and_purges_equal_boundary():
    cutoff = pd.Timestamp(gaussian.joint.CUTOFF)
    sessions = pd.DataFrame(dict(
        start_at=[cutoff - pd.Timedelta(hours=1)] * 3 + [cutoff, cutoff],
        terminal_known_at=[cutoff - pd.Timedelta(seconds=1), cutoff, cutoff + pd.Timedelta(minutes=1),
                           cutoff + pd.Timedelta(minutes=30), cutoff + pd.Timedelta(days=2)],
        invalid_total_cart=False, ambiguous_clock_sessions=False, is_censored=False))
    masks = gaussian.source_session_masks(sessions, cutoff + pd.Timedelta(days=1))
    assert masks["prefix"].tolist() == [True, False, False, False, False]
    assert masks["later"].tolist() == [False, False, False, True, False]
    assert masks["all"].tolist() == [True, True, True, True, False]
