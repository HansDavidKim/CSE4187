import numpy as np
from scipy import sparse

from simulate.v4.irl import fit_joint_reward_gmm as gmm
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl import reward_gmm


def test_batched_log_policy_matches_scalar_bellman_at_each_reward_draw():
    base = np.array([[.2, .7, -.4, .1], [-.4, 0., .6, -.1]])
    offsets = np.array([[.1, -.4, .3, .2, 0.], [-.5, .2, .5, -.7, .3]])
    mask = np.array([[True, True, False, True, False], [True] * 5])
    switch = sparse.csr_matrix([[0., 1.], [1., 0.]])
    matrices = [switch, sparse.eye(2, format="csr"), switch, switch, sparse.csr_matrix((2, 2))]
    transition = shopping.DISCOUNT * sparse.vstack(matrices, format="csr")

    batches = gmm.policy_logs(base, mask, transition, offsets).reshape(len(offsets), *mask.shape)
    for batch, vector in zip(batches, offsets, strict=True):
        expected, _, _ = shopping.soft_policy(shopping._reward(base + vector[:4]), matrices, mask)
        np.testing.assert_allclose(np.exp(batch[mask]), expected[mask], atol=2e-9, rtol=0.)
        np.testing.assert_array_equal(batch[~mask], 0.)


def test_joint_em_updates_weights_means_and_cross_covariances():
    points = np.r_[np.eye(5), -np.eye(5)]
    points[:, 1] += .4 * points[:, 0]
    sample = np.stack([points, points + 3.])
    responsibility = np.array([np.arange(1., 11.), np.ones(10)])

    weights, means, covariance = gmm.update_moments(sample, responsibility)

    np.testing.assert_allclose(weights, [55./65., 10./65.])
    for component, (points, mass) in enumerate(zip(sample, responsibility, strict=True)):
        np.testing.assert_allclose(means[component], np.average(points, weights=mass, axis=0))
        np.testing.assert_allclose(covariance[component], np.cov(points.T, aweights=mass, ddof=0), atol=1e-12)
    assert abs(covariance[1, 0, 1]) > .05


def test_runtime_newton_policy_matches_original_and_can_fall_back(monkeypatch):
    base = np.array([[.2, .7, -.4, .1], [-.4, 0., .6, -.1]])
    mask = np.array([[True, True, False, True, False], [True] * 5])
    switch = sparse.csr_matrix([[0., 1.], [1., 0.]])
    matrices = [switch, sparse.eye(2, format="csr"), switch, switch, sparse.csr_matrix((2, 2))]
    anchor = np.array([.1, -.4, .3, .2])
    _, anchor_value, _ = shopping.soft_policy(shopping._reward(base + anchor), matrices, mask)
    latent = {"common_reward": base, "action_mask": mask,
              "component_offsets": [anchor], "component_values": [anchor_value]}
    policy = reward_gmm.prepare_bellman(latent, matrices)
    offsets = np.array([2., -1., .1, -.3])
    expected, _, _ = shopping.soft_policy(shopping._reward(base + offsets), matrices, mask)

    np.testing.assert_allclose(policy(offsets), expected, atol=3e-10, rtol=0.)
    np.testing.assert_array_equal(policy(offsets)[~mask], 0.)

    monkeypatch.setattr(reward_gmm, "spsolve", lambda *_: np.full(len(base), np.nan))
    np.testing.assert_array_equal(policy(offsets), expected)


def test_electronics_gmm_uses_exact_observed_marginal_and_never_remove_pseudodata():
    from copy import deepcopy
    import pandas as pd
    from test_v4_latent_shopping import mdp
    available = ["view", "cart", "purchase"]
    active = [0, 1, 3, 4]
    means = np.array([[.1, .2, .3, -.1, .4], [-.3, .1, .5, .2, -.2]])
    covariance = np.repeat(np.eye(5)[None], 2, axis=0)
    covariance[:, 4, 0] = covariance[:, 0, 4] = .2
    nuisance_mean, nuisance_covariance = means.copy(), covariance.copy()
    nuisance_mean[:, 2] = [100000, -100000]
    nuisance_covariance[:, 2, :] = 7000
    nuisance_covariance[:, :, 2] = 7000
    normal = gmm.sobol_normal(3, 19, dimension=4)
    reference = gmm.draws(means, covariance, normal, available_actions=available)
    actual = gmm.draws(nuisance_mean, nuisance_covariance, normal, available_actions=available)
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_allclose(actual[:, :, active], means[:, None, active] + normal[None] @
                               np.linalg.cholesky(covariance[:, active][:, :, active]).swapaxes(1, 2))
    matrices, mask = mdp(); base = np.zeros((3, 4))
    transition = shopping.DISCOUNT * sparse.vstack(matrices, format="csr")
    counts = sparse.csr_matrix([[3., 2., 0., 0., 0., 0., 0., 0., 2., 1., 0., 0., 0., 0., 4.]])
    activity = pd.DataFrame(dict(base_logit=[.2], label=[1]))
    panel = (np.array(["u"]), counts, sparse.csr_matrix([[1.]]), activity)
    logs = gmm.policy_logs(base, mask, transition, reference.reshape(-1, 5))
    expected = gmm.joint_loss(panel, reference, np.array([.4, .6]), logs, moments=True)
    actual[:, :, 2] = 999999
    changed_logs = gmm.policy_logs(base, mask, transition, actual.reshape(-1, 5))
    candidate = gmm.joint_loss(panel, actual, np.array([.4, .6]), changed_logs, moments=True)
    assert expected[0] == candidate[0]
    np.testing.assert_array_equal(expected[1], candidate[1])
    a = gmm.update_moments(reference, expected[1], available_actions=available)
    b = gmm.update_moments(actual, candidate[1], available_actions=available)
    for before, after in zip(a, b):
        np.testing.assert_array_equal(before, after)
    block = dict(schema_version=1, type=gmm.TYPE, purpose="synthesis_only", parameter_names=gmm.PARAMETERS,
        component_count=2, mixture_weights=[.4, .6], means=means.tolist(), covariances=covariance.tolist(),
        base_model_sha256="a" * 64, return_model_sha256="b" * 64,
        available_actions=available, fitted_parameter_names=["view", "cart", "purchase", "return_logit"])
    changed = deepcopy(block)
    changed.update(means=nuisance_mean.tolist(), covariances=nuisance_covariance.tolist())
    factors = reward_gmm.validate_gmm(block, "a" * 64, "b" * 64, available_actions=available)
    changed_factors = reward_gmm.validate_gmm(changed, "a" * 64, "b" * 64, available_actions=available)
    solve = lambda offsets: shopping.soft_policy(shopping._reward(base + offsets), matrices, mask)[0]
    first = reward_gmm.sample_reward(block, factors, solve, "u", np.random.default_rng(12))[2]
    second = reward_gmm.sample_reward(changed, changed_factors, solve, "u", np.random.default_rng(12))[2]
    assert first["return_offset"] == second["return_offset"]
    np.testing.assert_array_equal(first["policy"], second["policy"])


def test_prepared_gmm_calibration_reserves_outer_users_and_all_postfit_outcomes():
    import hashlib
    import pandas as pd
    import pytest
    cutoff = pd.Timestamp("2021-02-01", tz="UTC")
    def role(uid):
        return (hashlib.sha256(f"42:{uid}".encode()).digest()[0] < 205,
                hashlib.sha256(f"42:inner:{uid}".encode()).digest()[0] < 205)
    train, selection, external = [next(str(uid) for uid in range(1000) if role(str(uid)) == target)
                                  for target in ((True, True), (True, False), (False, True))]
    sessions, activity = [], []
    for uid in (train, selection, external):
        for delta in (-1, 0, 1):
            known = cutoff + pd.Timedelta(seconds=delta)
            sessions.append(dict(user_id=uid, terminal_known_at=known, start_at=known - pd.Timedelta(minutes=30),
                                 invalid_total_cart=False, ambiguous_clock_sessions=False, is_censored=False))
            activity.append(dict(user_id=uid, target_at=known, at=known - pd.Timedelta(days=30),
                                 train_user=role(uid)[0], label=1))
    sessions, activity = pd.DataFrame(sessions), pd.DataFrame(activity)
    model = {"seed": 42, "source_contract": {"training_cutoff_at": cutoff.isoformat()},
             "observation_end": (cutoff + pd.Timedelta(days=90)).isoformat()}
    masks, audit = gmm.prepared_calibration_masks(sessions, activity, model, cutoff)
    for name, uid in (("train", train), ("selection", selection)):
        chosen = sessions.loc[masks[f"{name}_sessions"]]
        assert chosen.user_id.tolist() == [uid] and chosen.terminal_known_at.lt(cutoff).all()
        chosen = activity.loc[masks[f"{name}_activity"]]
        assert chosen.user_id.tolist() == [uid] and chosen.target_at.lt(cutoff).all()
    assert audit["outer_holdout_users_excluded"] == 1
    assert not audit["outer_holdout_and_postcutoff_labels_used_for_selection"]
    for key in ("donor_sessions", "donor_activity"):
        assert int(masks[key].sum()) == 2
    changed = activity.copy()
    changed.loc[changed.user_id.eq(external) | changed.target_at.ge(cutoff), "label"] = 0
    candidate, _ = gmm.prepared_calibration_masks(sessions, changed, model, cutoff)
    for name in masks:
        pd.testing.assert_series_equal(masks[name], candidate[name])
    model["source_contract"]["training_cutoff_at"] = (cutoff - pd.Timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="cutoff"):
        gmm.prepared_calibration_masks(sessions, activity, model, cutoff)
