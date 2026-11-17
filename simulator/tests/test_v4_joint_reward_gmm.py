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
