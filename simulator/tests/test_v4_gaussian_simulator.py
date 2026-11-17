"""Continuous customer rewards remain private and preserve the discrete fallback."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_v4_checkout_simulator import checkout_fixture, _customers
from simulate.v4.features import build_features
from simulate.v4.irl import fit_joint_latent_shopping as joint
from simulate.v4.irl import fit_latent_shopping as latent
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.paired_evaluation import paired_retention_rollouts
from simulate.v4.simulator.raw_store import RAW_COLUMNS, validate_raw_dataset


@pytest.fixture
def gaussian_fixture(checkout_fixture, tmp_path):
    config, model = copy.deepcopy(checkout_fixture)
    n, width = 8, len(latent.FEATURE_NAMES)
    encoder = dict(feature_names=latent.FEATURE_NAMES, state_keys=[[0, i] for i in range(n)],
                   context_means=np.zeros((n, width)).tolist(), feature_scale=[1.] * width,
                   tree=dict(left=[-1], right=[-1], feature=[-2], threshold=[-2.]))
    mask = latent.action_mask(encoder)
    matrices = [sparse.csr_matrix((mask[:, action].astype(float), (np.arange(n), np.zeros(n, dtype=int))),
                                 shape=(n, n)) for action in range(4)] + [sparse.csr_matrix((n, n))]
    base = np.full((n, 4), -4.)
    offsets = np.array([[-.3, 0., 0., .3], [-.1, 0., 0., .1],
                        [.1, 0., 0., -.1], [.3, 0., 0., -.3]])
    statistics = [latent.soft_policy(latent._reward(base + value), matrices, mask) for value in offsets]
    return_path = config['source_simulation']['return_model_path']
    return_sha = hashlib.sha256(Path(return_path).read_bytes()).hexdigest()
    return_offsets = [-.15, -.05, .05, .15]
    block = dict(schema_version=1, type=latent.TYPE, actions=list(latent.ACTIONS), discount=latent.DISCOUNT,
                 regularization=.01, state_count=n, feature_names=latent.FEATURE_NAMES,
                 encoder=encoder, action_mask=mask.tolist(), transition_csr=[latent.pack_matrix(p) for p in matrices],
                 mixture_weights=[.25] * 4, component_count=4, common_reward=base.tolist(),
                 component_offsets=offsets.tolist(), component_policies=[s[0].tolist() for s in statistics],
                 component_values=[s[1].tolist() for s in statistics],
                 profile_posteriors={p['profile_id']: [1., 0., 0., 0.] for p in model['profiles']},
                 return_activity=dict(schema_version=1, type=joint.ACTIVITY_TYPE, horizon_days=30,
                     base_model_sha256=return_sha, component_logit_offsets=return_offsets,
                     cutoff_at=pd.Timestamp(joint.CUTOFF).isoformat(),
                     fit_label_available_rule='window_end < cutoff',
                     posterior_available_rule='window_end < decision_clock'))
    model['latent_shopping'] = block
    actor = tmp_path / 'actor.json'
    actor.write_text(json.dumps(model))
    config['source_simulation']['model_path'] = str(actor)
    config['source_simulation']['max_events_per_session'] = 8
    gaussian = dict(schema_version=1, type='source_map_gaussian_reward_offsets',
                    base_model_sha256=hashlib.sha256(actor.read_bytes()).hexdigest(), return_model_sha256=return_sha,
                    means=np.column_stack([offsets, return_offsets]).tolist(), variances=np.zeros((4, 5)).tolist(),
                    parameter_names=['view', 'cart', 'remove', 'purchase', 'return_logit'],
                    purpose='synthesis_only', component_count=4, mixture_weights=[.25] * 4,
                    fit=dict(scope='all_fully_observed_source_user_map', forecast_validation_uses_gaussian=False))
    return config, gaussian, tmp_path / 'gaussian.json'


def enable_gaussian(config, gaussian, path):
    path.write_text(json.dumps(gaussian))
    result = copy.deepcopy(config)
    result['source_simulation']['gaussian_reward_model_path'] = str(path)
    return result


def test_zero_variance_centroid_matches_discrete_tables_and_keeps_old_policy_call(gaussian_fixture, monkeypatch):
    config, gaussian, path = gaussian_fixture
    original_policy = latent.policy
    def old_policy(model, context, component):
        return original_policy(model, context, component)
    with monkeypatch.context() as legacy:
        legacy.setattr(latent, 'policy', old_policy)
        discrete = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    continuous = EcommerceSimulator(enable_gaussian(config, gaussian, path)).generate(
        months=3, seed=17, customers=_customers())
    for name in discrete.tables:
        pd.testing.assert_frame_equal(discrete.tables[name], continuous.tables[name][discrete.tables[name].columns])
    assert 'gaussian_reward' not in discrete.metadata
    assert discrete.metadata['simulator_version'] == 'source-shopping-v10'
    assert continuous.metadata['simulator_version'] == 'source-shopping-v11'


def test_gaussian_draw_is_continuous_persistent_reproducible_and_private(gaussian_fixture, monkeypatch, tmp_path):
    from simulate.v4.irl import fit_gaussian_rewards as gaussian_runtime
    config, gaussian, path = gaussian_fixture
    gaussian['variances'] = np.full((4, 5), .025).tolist()
    config = enable_gaussian(config, gaussian, path)
    original_sample, original_policy = gaussian_runtime.sample_reward_policy, latent.policy
    policies, observations = {}, []
    def draw(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        policies[id(result['policy'])] = result['policy'].copy()
        return result
    def policy(model, context, component, probabilities=None):
        assert probabilities is not None
        np.testing.assert_array_equal(probabilities, policies[id(probabilities)])
        observations.append(id(probabilities))
        return original_policy(model, context, component, probabilities)
    monkeypatch.setattr(gaussian_runtime, 'sample_reward_policy', draw)
    monkeypatch.setattr(latent, 'policy', policy)
    result = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    assert len(policies) == len(_customers()) and set(observations) == set(policies)
    private = [column for column in result.tables['diagnostics'] if column.startswith('gaussian_')]
    assert len(private) == 5
    assert result.tables['diagnostics'].latent_reward_component.eq(0).all()
    assert result.tables['diagnostics'][private].drop_duplicates().shape[0] == len(_customers())
    assert not set(private) & set(build_features(result.tables, as_of_at=pd.Timestamp(result.metadata['as_of_at'])))
    for name in RAW_COLUMNS:
        assert not set(private) & set(result.tables[name])
    validate_raw_dataset({name: result.tables[name] for name in RAW_COLUMNS}, result.metadata)
    pairs, _ = paired_retention_rollouts(result.tables['diagnostics'], result.metadata, repetitions=1)
    assert pairs.simulator_version.eq('source-shopping-v11').all()
    result.save(tmp_path / 'saved')
    pd.testing.assert_frame_equal(result.tables['diagnostics'], pd.read_parquet(tmp_path / 'saved/diagnostics.parquet'))
    policies.clear(); observations.clear()
    again = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    assert result.metadata == again.metadata
    for name in result.tables:
        pd.testing.assert_frame_equal(result.tables[name], again.tables[name])


@pytest.mark.parametrize('field', ['base_model_sha256', 'return_model_sha256'])
def test_gaussian_sidecar_requires_exact_frozen_models(gaussian_fixture, field):
    config, gaussian, path = gaussian_fixture
    gaussian[field] = '0' * 64
    with pytest.raises(ValueError):
        EcommerceSimulator(enable_gaussian(config, gaussian, path)).generate(months=3, customers=_customers())


def test_gaussian_sidecar_content_changes_generation_identity(gaussian_fixture):
    config, gaussian, path = gaussian_fixture
    config = enable_gaussian(config, gaussian, path)
    first = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    gaussian['means'][0][4] += .01
    path.write_text(json.dumps(gaussian))
    changed = EcommerceSimulator(config).generate(months=3, seed=17, customers=_customers())
    assert first.metadata['generation_fingerprint'] != changed.metadata['generation_fingerprint']


def test_learned_gmm_uses_its_own_components_and_private_persistent_offsets(gaussian_fixture, tmp_path):
    config, gaussian, _ = gaussian_fixture
    model = json.loads(Path(config['source_simulation']['model_path']).read_text())
    # Component 5 exists in the fitted GMM, outside the parent's four-policy mixture.
    gmm = dict(gaussian, type='source_map_reward_gmm', component_count=6,
               mixture_weights=[1 / 6] * 6, means=[[0.] * 5] * 6,
               covariances=(np.tile(np.eye(5), (6, 1, 1)) * .025).tolist(),
               profile_posteriors={p['profile_id']: [0.] * 5 + [1.] for p in model['profiles']})
    path = tmp_path / 'gmm.json'
    path.write_text(json.dumps(gmm))
    config['source_simulation']['reward_gmm_model_path'] = str(path)
    simulator = EcommerceSimulator(config)
    result = simulator.generate(months=3, seed=17, customers=_customers())
    repeat = simulator.generate(months=3, seed=17, customers=_customers())
    assert result.metadata == repeat.metadata
    assert result.metadata['simulator_version'] == 'source-shopping-v12'
    assert result.metadata['reward_gmm']['mixture_weights'] == gmm['mixture_weights']
    diagnostics = result.tables['diagnostics']
    assert diagnostics.latent_reward_component.eq(5).all()
    assert diagnostics.latent_assignment_scope.eq('gmm_source_donor_posterior').all()
    private = [column for column in diagnostics if column.startswith('gaussian_')]
    assert len(private) == 5 and diagnostics[private].drop_duplicates().shape[0] == len(_customers())
    assert not set(private) & set(build_features(result.tables, as_of_at=pd.Timestamp(result.metadata['as_of_at'])))
    for name in result.tables:
        pd.testing.assert_frame_equal(result.tables[name], repeat.tables[name])
        if name in RAW_COLUMNS:
            assert not set(private) & set(result.tables[name])
    pairs, _ = paired_retention_rollouts(diagnostics, result.metadata, repetitions=1)
    assert pairs.simulator_version.eq('source-shopping-v12').all()


def test_reward_gmm_rejects_singular_covariance(gaussian_fixture, tmp_path):
    config, gaussian, _ = gaussian_fixture
    gmm = dict(gaussian, type='source_joint_reward_gmm', covariances=np.zeros((4, 5, 5)).tolist())
    path = tmp_path / 'gmm.json'
    path.write_text(json.dumps(gmm))
    config['source_simulation']['reward_gmm_model_path'] = str(path)
    with pytest.raises(ValueError, match='positive definite'):
        EcommerceSimulator(config).generate(months=3, customers=_customers())
