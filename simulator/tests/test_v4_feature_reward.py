import json

import numpy as np
import pytest
from scipy import sparse

from simulate.v4.irl.feature_reward import feature_reward_objective, fit_feature_reward
from simulate.v4.irl.fit_latent_shopping import soft_policy


def example():
    matrices = [sparse.csr_matrix((3, 3)) for _ in range(5)]
    for action, source, target in [(0, 0, 1), (1, 0, 2), (3, 1, 2)]:
        p = matrices[action].tolil()
        p[source, target] = 1
        matrices[action] = p.tocsr()
    mask = np.zeros((3, 5), dtype=bool)
    mask[0, :2] = True
    mask[1, [3, 4]] = True
    mask[2, 4] = True
    counts = np.zeros((3, 5))
    counts[0, :2] = [15, 3]
    counts[1, [3, 4]] = [10, 5]
    counts[2, 4] = 13
    return np.array([[1., -1.], [1., .5], [1., 2.]]), counts, matrices, mask


def test_feature_chain_gradient_matches_finite_difference_through_bellman():
    inputs = example()
    theta = np.linspace(-.3, .4, 8)
    _, gradient = feature_reward_objective(theta, *inputs, discount=.9, regularization=.03)
    for i in range(len(theta)):
        change = np.zeros_like(theta)
        change[i] = 1e-5
        high = feature_reward_objective(theta + change, *inputs, discount=.9, regularization=.03)[0]
        low = feature_reward_objective(theta - change, *inputs, discount=.9, regularization=.03)[0]
        assert gradient[i] == pytest.approx((high - low) / 2e-5, abs=2e-6)


def test_fit_has_linear_reward_zero_end_and_exact_bellman_selfcheck():
    basis, counts, matrices, mask = example()
    original = [matrix.copy() for matrix in matrices]
    result = fit_feature_reward(basis, counts, matrices, mask, feature_names=['bias', 'past_feature'])
    weights, reward = np.array(result['reward_weights']), np.array(result['common_reward'])
    assert weights.shape == (4, 2)  # Not a free3-by-action table.
    np.testing.assert_allclose(reward, basis @ weights.T)
    full_reward = np.column_stack([reward, np.zeros(3)])
    probability, value, residual = soft_policy(full_reward, matrices, mask, result['discount'])
    np.testing.assert_allclose(result['policy'], probability, atol=1e-12)
    np.testing.assert_allclose(result['value'], value, atol=1e-12)
    np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-12)
    assert not probability[~mask].any()
    assert residual < 1e-9
    assert result['fit']['converged']
    trace = result['fit']['trace']
    assert [row['iteration'] for row in trace] == list(range(result['fit']['iterations'] + 1))
    assert trace[-1]['objective'] < trace[0]['objective']
    assert trace[-1]['raw_nll'] == result['fit']['raw_nll']
    assert result['fit']['parameter_count'] == 8
    json.dumps(result, allow_nan=False)
    for before, after in zip(original, matrices):
        assert (before != after).nnz == 0


def test_iteration_cap_is_not_reported_as_convergence():
    result = fit_feature_reward(*example(), feature_names=['bias', 'past'], max_iter=1)
    assert not result['fit']['converged']
    assert result['fit']['iterations'] == 1
    assert len(result['fit']['trace']) == 2


def test_fixed_reference_offset_gradient_and_zero_reward_recover_markov_policy():
    basis, counts, matrices, mask = example()
    reference = counts / counts.sum(axis=1, keepdims=True)
    offset = np.zeros_like(reference)
    offset[mask] = np.log(reference[mask])
    probability, value, residual = soft_policy(offset, matrices, mask, .9)
    np.testing.assert_allclose(probability, reference, atol=1e-12)
    np.testing.assert_allclose(value, 0, atol=1e-12)
    assert residual < 1e-9
    theta = np.linspace(-.3, .4, 8)
    options = dict(discount=.9, regularization=.03, reward_offset=offset)
    _, gradient = feature_reward_objective(theta, basis, counts, matrices, mask, **options)
    for i in range(len(theta)):
        change = np.zeros_like(theta);change[i] = 1e-5
        high = feature_reward_objective(theta + change, basis, counts, matrices, mask, **options)[0]
        low = feature_reward_objective(theta - change, basis, counts, matrices, mask, **options)[0]
        assert gradient[i] == pytest.approx((high-low)/2e-5, abs=2e-6)
    result = fit_feature_reward(basis, counts, matrices, mask,
        feature_names=['bias', 'past'], reward_offset=offset)
    reward = np.column_stack([result['common_reward'], np.zeros(len(basis))]) + offset
    policy, _, _ = soft_policy(reward, matrices, mask, result['discount'])
    np.testing.assert_allclose(result['policy'], policy, atol=1e-12)
    np.testing.assert_array_equal(result['fixed_reward_offset'], offset)
    assert result['fit']['converged']


@pytest.mark.parametrize('offset', [np.zeros((2, 5)), np.full((3, 5), np.nan),
                                     np.full((3, 5), -np.inf)])
def test_invalid_fixed_offset_is_rejected(offset):
    with pytest.raises(ValueError, match='offset'):
        fit_feature_reward(*example(), feature_names=['bias', 'past'], reward_offset=offset)


def test_zero_basis_column_keeps_unobserved_feature_coefficients_zero():
    basis, counts, matrices, mask = example()
    basis = np.column_stack([basis, np.zeros(len(basis))])
    result = fit_feature_reward(basis, counts, matrices, mask,
        feature_names=['bias', 'past', 'unobserved_source_field'])
    assert result['fit']['converged']
    np.testing.assert_array_equal(np.asarray(result['reward_weights'])[:, 2], 0.)


def test_persona_centered_map_penalty_has_correct_gradient_and_pulls_to_prior():
    inputs = example()
    prior = np.arange(8, dtype=float).reshape(4, 2) / 10
    theta = np.linspace(-.3, .4, 8)
    _, gradient = feature_reward_objective(theta, *inputs, regularization=.7, prior=prior)
    for i in range(len(theta)):
        change = np.zeros_like(theta);change[i] = 1e-5
        high = feature_reward_objective(theta + change, *inputs, regularization=.7, prior=prior)[0]
        low = feature_reward_objective(theta - change, *inputs, regularization=.7, prior=prior)[0]
        assert gradient[i] == pytest.approx((high-low)/2e-5, abs=2e-6)
    result = fit_feature_reward(*inputs, feature_names=['bias', 'past'], regularization=100., prior=prior,
                                initial=prior)
    assert result['fit']['converged']
    assert np.max(np.abs(np.asarray(result['reward_weights'])-prior)) < .02


@pytest.mark.parametrize('prior', [np.zeros(8), np.full((4, 2), np.nan), np.zeros((3, 2))])
def test_invalid_persona_prior_is_rejected(prior):
    with pytest.raises(ValueError, match='prior'):
        fit_feature_reward(*example(), feature_names=['bias', 'past'], prior=prior)


@pytest.mark.parametrize('defect', ['nan_basis', 'negative_count', 'zero_count', 'masked_count',
    'integer_mask', 'empty_mask_row', 'bad_transition_shape', 'negative_transition',
    'nan_transition', 'missing_transition_mass', 'nonzero_end'])
def test_rejects_invalid_counts_masks_and_sparse_dynamics(defect):
    basis, counts, matrices, mask = example()
    if defect == 'nan_basis': basis[0, 0] = np.nan
    elif defect == 'negative_count': counts[0, 0] = -1
    elif defect == 'zero_count': counts[:] = 0
    elif defect == 'masked_count': counts[0, 4] = 1
    elif defect == 'integer_mask': mask = mask.astype(int)
    elif defect == 'empty_mask_row': mask[2] = False
    elif defect == 'bad_transition_shape': matrices[0] = sparse.eye(2)
    elif defect == 'negative_transition': matrices[0].data[0] = -1
    elif defect == 'nan_transition': matrices[0].data[0] = np.nan
    elif defect == 'missing_transition_mass': matrices[0].data[0] = .5
    elif defect == 'nonzero_end': matrices[4] = sparse.eye(3)
    with pytest.raises(ValueError):
        fit_feature_reward(basis, counts, matrices, mask, feature_names=['bias', 'past'])


@pytest.mark.parametrize('options', [dict(feature_names=['same', 'same']), dict(feature_names=['only']),
    dict(feature_names=['bias', 'past'], max_iter=True),
    dict(feature_names=['bias', 'past'], discount=1),
    dict(feature_names=['bias', 'past'], regularization=-1),
    dict(feature_names=['bias', 'past'], initial=np.zeros(3)),
    dict(feature_names=['bias', 'past'], initial=np.full((4, 2), np.inf))])
def test_rejects_invalid_feature_contract_and_fit_options(options):
    with pytest.raises(ValueError):
        fit_feature_reward(*example(), **options)
