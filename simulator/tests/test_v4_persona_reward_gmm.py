from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from simulate.v4.irl import fit_latent_shopping as shopping, persona_reward_gmm as persona, reward_gmm
from test_v4_latent_shopping import mdp


AVAILABLE = ["view", "cart", "purchase"]
CUTOFF = pd.Timestamp("2020-01-29", tz="UTC")


def actor():
    matrices, mask = mdp()
    base = np.arange(12).reshape(3, 4) * .05
    base[:, 2] = 0.
    deviation = np.zeros((2, 3, 4))
    deviation[0, :, 0], deviation[1, :, 0] = -.4, .4
    solved = [shopping.soft_policy(shopping._reward(base + d), matrices, mask) for d in deviation]
    latent = dict(common_reward=base.tolist(), component_reward_deviations=deviation.tolist(),
        component_offsets=np.zeros((2, 4)).tolist(), component_values=[x[1].tolist() for x in solved],
        component_policies=[x[0].tolist() for x in solved], action_mask=mask.tolist(), discount=.95,
        encoder={"available_actions": AVAILABLE}, transition_csr=[shopping.pack_matrix(p) for p in matrices])
    return latent, matrices


def block():
    return dict(schema_version=1, type=persona.TYPE, purpose="synthesis_only", parameter_names=persona.PARAMETERS,
        component_count=2, component_model_indices=[1, 0], persona_labels=["A", "B"],
        mixture_weights=[.3, .7], means=[[.2, 0., 0., .3, .4], [-.1, .2, 0., 0., -.2]],
        covariances=[(np.eye(5) * .01).tolist()] * 2, available_actions=AVAILABLE,
        fitted_parameter_names=["view", "cart", "purchase", "return_logit"],
        base_model_sha256="a" * 64, return_model_sha256="b" * 64)


def _source():
    def role(uid):
        return (hashlib.sha256(f"42:{uid}".encode()).digest()[0] < 205,
                hashlib.sha256(f"42:inner:{uid}".encode()).digest()[0] < 205)
    train = [str(uid) for uid in range(1000) if role(str(uid)) == (True, True)]
    validation = [str(uid) for uid in range(1000) if role(str(uid)) == (True, False)]
    external = next(str(uid) for uid in range(1000) if role(str(uid)) == (False, True))
    assignments, actions, activity = [], [], []
    for group, start in (("A", 0), ("B", 2)):
        users = [*train[start:start+2], *validation[start:start+2]]
        for i, uid in enumerate(users):
            assignments.append(dict(user_id=uid, cluster=group, assignment_cutoff=CUTOFF,
                                    representation_end_at=CUTOFF - pd.Timedelta(seconds=1)))
            if i % 2 == 0:
                actions.append(dict(user_id=uid, state=0, action=0, known_at=CUTOFF - pd.Timedelta(days=5)))
            at = CUTOFF - pd.Timedelta(days=45)
            activity.append(dict(user_id=uid, at=at, target_at=at + pd.Timedelta(days=30), base_logit=0., label=i % 2))
    for uid, group in ((train[4], "A"), (external, "B")):
        assignments.append(dict(user_id=uid, cluster=group, assignment_cutoff=CUTOFF,
                                representation_end_at=CUTOFF - pd.Timedelta(seconds=1)))
    for uid in (train[0], external):
        activity.append(dict(user_id=uid, at=CUTOFF, target_at=CUTOFF + pd.Timedelta(days=30), base_logit=0., label=1))
    return pd.DataFrame(actions), pd.DataFrame(activity), pd.DataFrame(assignments), external


def _fit(actions, activity, assignments):
    latent, _ = actor()
    return persona.fit_persona_reward_gmm(latent, actions, activity, assignments,
        component_model_indices={"A": 0, "B": 1}, cutoff_at=CUTOFF,
        max_updates=1, qmc_power=3, condition_on_profile_components=True,
        base_model_sha256="a" * 64, return_model_sha256="b" * 64)


def test_persona_solver_scales_both_actor_slope_and_personal_offset():
    latent, matrices = actor()
    solve = reward_gmm.prepare_bellman(latent, matrices)
    offset = np.array([.2, -.1, 0., .3])
    for z in range(2):
        expected, _, _ = shopping.soft_policy(shopping._reward(
            (np.asarray(latent["common_reward"]) + latent["component_reward_deviations"][z] + offset) / 2),
            matrices, np.asarray(latent["action_mask"]))
        np.testing.assert_allclose(solve(offset, component_id=z, temperature=2), expected, atol=3e-10, rtol=0)
    with pytest.raises(ValueError, match="actor index"):
        solve(offset)


def test_runtime_keeps_raw_covariance_at_zero_dispersion_and_maps_profile_persona():
    latent, matrices = actor(); model = block()
    model.update(dispersion_scales=[0., 0.], choice_temperatures=[2., 3.], condition_on_profile_components=True)
    before = json.dumps(model)
    factors = reward_gmm.validate_gmm(model, "a" * 64, "b" * 64, available_actions=AVAILABLE)
    solve = reward_gmm.prepare_bellman(latent, matrices)
    z, scope, draw = reward_gmm.sample_reward(model, factors, solve, "p", np.random.default_rng(4), profile_components=[1., 0.])
    assert z == 1 and scope == "gmm_profile_persona_prior"
    assert draw["return_offset"] == -.2
    np.testing.assert_array_equal(draw["action_offsets"], model["means"][1][:4])
    np.testing.assert_allclose(draw["policy"], solve(np.asarray(model["means"][1][:4]), component_id=0, temperature=3), atol=1e-12)
    assert json.dumps(model) == before
    invalid = deepcopy(model); invalid["component_model_indices"] = [0, 0]
    with pytest.raises(ValueError, match="one-to-one"):
        reward_gmm.validate_gmm(invalid, "a" * 64, "b" * 64)
    invalid = deepcopy(model); invalid["choice_temperatures"][0] = 0.
    with pytest.raises(ValueError, match="temperature"):
        reward_gmm.validate_gmm(invalid, "a" * 64, "b" * 64)


def test_known_persona_fit_counts_return_only_users_and_full_prefix_prior():
    actions, activity, assignments, _ = _source()
    model = _fit(actions, activity, assignments)
    assert model["persona_labels"] == ["A", "B"] and model["component_model_indices"] == [0, 1]
    np.testing.assert_allclose(model["mixture_weights"], [5/9, 4/9])
    for stats in model["fit"]["personas"].values():
        assert stats["training_users"] == stats["selection_users"] == 2
        assert stats["training_return_only_users"] == stats["selection_return_only_users"] == 1
    assert not model["fit"]["persona_groups_refitted_or_reclustered"]
    factors = reward_gmm.validate_gmm(model, "a" * 64, "b" * 64, available_actions=AVAILABLE)
    assert factors.shape == (2, 4, 4)
    assert np.linalg.eigvalsh(model["covariances"]).min() > 0


def test_outer_users_and_future_activity_cannot_change_persona_fit_or_selection():
    actions, activity, assignments, external = _source()
    expected = _fit(actions, activity, assignments)
    changed = activity.copy()
    changed.loc[changed.user_id.eq(external) | changed.target_at.ge(CUTOFF), "label"] = 0
    actual = _fit(actions, changed, assignments)
    assert expected == actual


def test_return_only_likelihood_integrates_missing_shopping_instead_of_zero_map():
    latent, matrices = actor()
    actions = pd.DataFrame(columns=["user_id", "state", "action", "count"])
    activity = pd.DataFrame([dict(user_id="x", base_logit=.2, label=0)])
    panel = persona._panel(actions, activity, {"x"}, 3)
    assert panel["counts"].nnz == 0 and panel["return_only_users"] == 1
    sample = np.array([[0., 0., 0., 0., -.5], [1., 2., 0., 3., .5]])
    transition = shopping.DISCOUNT * sparse.vstack(matrices, format="csr")
    first = persona._likelihood(panel, sample, np.zeros((3, 4)), np.asarray(latent["action_mask"]), transition)
    sample[:, :4] = 10000
    second = persona._likelihood(panel, sample, np.zeros((3, 4)), np.asarray(latent["action_mask"]), transition)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])


def test_invalid_future_persona_prefix_and_unsupported_action_are_rejected():
    actions, activity, assignments, _ = _source()
    bad = assignments.copy(); bad.loc[0, "assignment_cutoff"] = CUTOFF + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="frozen prefix"):
        _fit(actions, activity, bad)
    bad = actions.copy(); bad.loc[0, "action"] = 2
    with pytest.raises(ValueError, match="support"):
        _fit(bad, activity, assignments)


def test_daily_provider_fits_return_through_joint_day_outcomes_without_monthly_labels():
    from scipy.special import logsumexp
    _, activity, assignments, external = _source()
    observed = [uid for uid in activity.user_id.unique() if uid != external]
    actions = pd.DataFrame([dict(user_id=uid, state=0, action=i % 4,
                                 known_at=CUTOFF - pd.Timedelta(days=5)) for i, uid in enumerate(observed)])
    descriptor = dict(state_count=1, component_count=2, action_mask=[[True] * 4 + [False]])
    incidence = np.array([[0., 0., 0., 0., 0.], [1., 0., 0., 0., 1.],
                          [0., 0., 0., 1., 1.], [1., 0., 0., 1., 1.]])
    def provider(index, sample):
        logits = sample @ incidence.T + np.array([0., .2 * index, -.1 * index, .1])
        result = np.zeros((len(sample), 5))
        result[:, :4] = logits - logsumexp(logits, axis=1, keepdims=True)
        return result
    model = persona.fit_persona_reward_gmm(descriptor, actions, pd.DataFrame(), assignments,
        component_model_indices={"A": 0, "B": 1}, cutoff_at=CUTOFF,
        max_updates=1, qmc_power=3, backend="tianchi_daily", active_parameter_indices=[0, 3, 4],
        policy_log_provider=provider)
    assert model["type"] == persona.DAILY_TYPE and model["backend"] == "tianchi_daily"
    assert model["parameter_names"][-1] == "return"
    assert model["component_fitted_parameter_names"] == [["view", "purchase", "return"]] * 2
    with pytest.raises(ValueError, match="incompatible"):
        reward_gmm.validate_gmm(model, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="daily return evidence"):
        persona.fit_persona_reward_gmm(descriptor, actions, activity, assignments,
            component_model_indices={"A": 0, "B": 1}, cutoff_at=CUTOFF,
            backend="tianchi_daily", active_parameter_indices=[0, 3, 4], policy_log_provider=provider)


def test_retail_missing_activity_is_not_reinterpreted_as_daily_return_evidence():
    actions, activity, assignments, _ = _source(); latent, _ = actor()
    with pytest.raises(ValueError, match="retail requires"):
        persona.fit_persona_reward_gmm(latent, actions, activity.iloc[:0], assignments,
            component_model_indices={"A": 0, "B": 1}, cutoff_at=CUTOFF)


def test_action_only_static_return_particles_use_no_activity_posterior_and_bound_memory():
    actions, activity, assignments, _ = _source(); latent, _ = actor(); model = block()
    model["covariances"] = np.repeat(np.eye(5)[None] * .04, 2, axis=0).tolist()
    model["profile_posteriors"] = {uid: [.99, .01] for uid in assignments.user_id}
    priors = persona.action_only_static_return_prior(latent, model, actions, assignments,
                                                    cutoff_at=CUTOFF, qmc_power=3, batch_users=1)
    expected = persona.action_only_static_return_prior(latent, {k: v for k, v in model.items() if k != "profile_posteriors"},
        actions, assignments, cutoff_at=CUTOFF, qmc_power=3, batch_users=256)
    for uid in priors:
        np.testing.assert_array_equal(priors[uid]["offsets"], expected[uid]["offsets"])
        np.testing.assert_allclose(priors[uid]["log_weights"], expected[uid]["log_weights"], atol=1e-14, rtol=0)
        assert priors[uid]["posterior_uses_activity_labels"] is False
    same_group = assignments.loc[assignments.cluster.eq("A"), "user_id"].tolist()
    assert priors[same_group[0]]["offsets"] is priors[same_group[1]]["offsets"]
    no_action = next(uid for uid in assignments.user_id if uid not in set(actions.user_id))
    np.testing.assert_allclose(priors[no_action]["log_weights"], -np.log(8), atol=1e-14)
    future = pd.concat([actions, pd.DataFrame([dict(user_id=no_action, state=0, action=0,
                                                   known_at=CUTOFF + pd.Timedelta(days=1))])])
    after = persona.action_only_static_return_prior(latent, model, future, assignments,
                                                   cutoff_at=CUTOFF, qmc_power=3)
    for uid in priors:
        np.testing.assert_array_equal(priors[uid]["log_weights"], after[uid]["log_weights"])
