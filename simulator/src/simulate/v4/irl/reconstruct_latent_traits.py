"""Reconstruct donor traits from observed source histories for synthesis only.

This performs an empirical-Bayes E step with frozen shopping/return parameters.
It never refits a reward, transition, encoder, mixture prior or forecast score.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from simulate.v4.irl import fit_joint_latent_shopping as joint
from simulate.v4.irl import fit_latent_shopping as shopping


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_trait_posteriors(rows, model, return_model):
    """Use all supported complete source sessions and matured activity windows."""
    block = model['latent_shopping']
    data = shopping.build_source(rows, model['observation_end'], int(model.get('seed', 42)))
    sessions = data['sessions']
    valid = ~sessions.invalid_total_cart & ~sessions.ambiguous_clock_sessions & ~sessions.is_censored
    valid &= sessions.terminal_known_at.le(pd.Timestamp(model['observation_end']))
    panel = shopping.select_panel(data, valid)
    state, fallback = shopping.encode(block['encoder'], panel['features'])
    mask = np.asarray(block['action_mask'], dtype=bool)
    if not mask[state, panel['records'].action.to_numpy()].all():
        raise ValueError('fully supported source sessions contain an impossible fitted action')
    activity, activity_audit = joint.build_activity_panel(rows, return_model, int(model.get('seed', 42)))
    activity = activity.loc[activity.target_at.le(pd.Timestamp(model['observation_end']))].reset_index(drop=True)
    users, counts, incidence, _ = joint.joint_counts(panel, state, len(mask), activity)
    probabilities = np.asarray(block['component_policies'])
    log_policy = np.zeros_like(probabilities)
    log_policy[:, mask] = np.log(np.maximum(probabilities[:, mask], 1e-300))
    likelihood = counts @ log_policy.reshape(len(probabilities), -1).T
    offsets = np.asarray(block['return_activity']['component_logit_offsets'])
    likelihood += incidence @ joint.activity_log_probability(activity, offsets)
    log_joint = likelihood + np.log(np.maximum(block['mixture_weights'], 1e-300))[None, :]
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    bank = {hashlib.sha256(f'cosmetics:{uid}'.encode()).hexdigest()[:16]: q.tolist()
            for uid, q in zip(users, posterior)}
    if len(bank) != len(users) or not np.isfinite(posterior).all() or not np.allclose(posterior.sum(axis=1), 1):
        raise ValueError('source trait posterior identities or probabilities differ')
    return bank, {
        'source_rows': len(rows), 'supported_sessions': int(valid.sum()),
        'excluded_sessions': int((~valid).sum()), 'supported_action_end_observations': len(panel['records']),
        'fully_observed_activity_observations': len(activity), 'activity_users': int(activity.user_id.nunique()),
        'activity_positive': int(activity.label.sum()), 'activity_panel': activity_audit,
        'encoder_fallback_observations': int(fallback.sum()), 'posterior_profiles': len(bank),
        'effective_customer_mass': posterior.sum(axis=0).tolist(),
        'posterior_entropy_mean': float(-(posterior * np.log(np.maximum(posterior, 1e-300))).sum(axis=1).mean()),
        'posterior_max_probability_mean': float(posterior.max(axis=1).mean()),
        'source_audit': data['audit'],
        'likelihood_rule': 'sum of supported source action/END log policies plus fully observed non-entry 30d activity log probabilities; one source user per prior draw',
        'support_rule': 'known-total valid, unambiguous, fully observed whole sessions; no unsupported source row stitching; original known-SKU ledger unchanged',
        'availability_rule': 'END target last_action+30m <= source observation_end; activity window_end <= source observation_end; synthesis after the source observation period only',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('data/models/shopping_joint_latent_irl_candidate.json'))
    parser.add_argument('--source-cache', type=Path, default=Path('artifacts/v4/source_sample.parquet'))
    parser.add_argument('--output', type=Path, default=Path('data/models/shopping_joint_latent_source_traits.json'))
    parser.add_argument('--report', type=Path, default=Path('docs/audit/simulator/joint_latent_source_traits.json'))
    args = parser.parse_args()
    if len({args.base.resolve(), args.output.resolve(), args.report.resolve()}) != 3:
        raise ValueError('parent model, synthesis model and report paths must differ')
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f'immutable reconstruction output exists: {path}')
    parent = json.loads(args.base.read_text())
    block = parent['latent_shopping']
    if block.get('profile_posterior_assignment_scope') is not None:
        raise ValueError('reconstruction requires the original strict-prefix parent artifact')
    return_path = Path(block['provenance']['return_base_path'])
    return_sha = digest(return_path)
    joint.validate_model(parent, return_sha)
    return_model = json.loads(return_path.read_text())
    from simulate.v4.irl.fit_donor_prior import load_prior_source
    rows, descriptor = load_prior_source(parent, args.source_cache)
    before = shopping.trait_invariant_sha256(block)
    bank, evidence = source_trait_posteriors(rows, parent, return_model)
    result = deepcopy(parent)
    target = result['latent_shopping']
    target['profile_posteriors'] = bank
    target['profile_posterior_assignment_scope'] = 'source_observation_trait_posterior'
    target['posterior_bank_scope'] = 'synthesis only: all fully observed supported source actions and activity outcomes; no live forecast or heldout prequential posterior use'
    target['runtime_latent'] = 'one ex ante private donor-trait draw from the full-source empirical-Bayes posterior, else unchanged population prior; not the generated customer future and not a predictor'
    covered = len({profile['profile_id'] for profile in parent['profiles']} & bank.keys())
    target['trait_reconstruction'] = {
        'schema_version': 1, 'purpose': 'synthesis_only',
        'parent_model_path': str(args.base), 'parent_model_sha256': digest(args.base),
        'source_cache_path': str(args.source_cache), 'source_cache_sha256': digest(args.source_cache),
        'source_descriptor': descriptor, 'reconstructor_sha256': digest(__file__),
        'shopping_runtime_sha256': digest(shopping.__file__), 'joint_fitter_sha256': digest(joint.__file__),
        'return_base_path': str(return_path), 'return_base_sha256': return_sha,
        'immutable_parent_block_sha256': before,
        'observation_start_at': pd.Timestamp(parent['observation_start']).isoformat(),
        'observation_end_at': pd.Timestamp(parent['observation_end']).isoformat(),
        'posterior_profiles': len(bank), 'generator_profiles_covered': covered,
        'generator_profiles_using_population_prior': len(parent['profiles']) - covered,
        'forecast_validation_uses_bank': False,
        'forecast_validation_contract': 'parent training/strict prequential validation remains byte-value identical; this all-source bank must never supply posterior weights to a forecast evaluation',
        'assumption': 'Population reconstruction conditional on observed donor histories. Reference-source fidelity is not independent population generalization and not same-customer future prediction.',
        **evidence,
    }
    if shopping.trait_invariant_sha256(target) != before:
        raise ValueError('source reconstruction modified the frozen parent parameters or forecast validation')
    joint.validate_model(result, return_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    common = set(bank) & block['profile_posteriors'].keys()
    distance = np.array([np.abs(np.asarray(bank[key])-block['profile_posteriors'][key]).sum() for key in sorted(common)])
    report = {
        'passed': True, 'purpose': 'synthesis-only source donor posterior reconstruction; not training or forecast validation',
        'model_path': str(args.output), 'model_sha256': digest(args.output),
        'parent_model_sha256': digest(args.base), 'frozen_parent_block_sha256': before,
        'parameters_and_forecast_validation_unchanged': True,
        'source_and_checkout_catalog_unchanged': all(result[key] == value for key, value in parent.items() if key != 'latent_shopping'),
        'parent_posterior_profiles': len(block['profile_posteriors']), 'posterior_profiles': len(bank),
        'generator_profiles_covered': covered, 'shared_profiles': len(common),
        'shared_profile_posterior_L1_mean': float(distance.mean()) if len(distance) else None,
        'reconstruction': target['trait_reconstruction'],
        'adoption': 'candidate; benchmark this explicit source-conditioned synthesis regime before selecting a simulator default',
        'limitations': ['Full-source posterior includes observations outside the parent training split. It is forbidden as a posterior for heldout/live forecasting.',
            'This is a conditional population reconstruction, comparable to a source donor scenario; it does not establish performance on unseen real people.',
            'Shopping reward, transition model, encoder, component prior, activity coefficients and original validation are not refitted.',
            'Exact generated cart physics and source unknown-SKU/left-censoring limitations remain.'],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ('model_path', 'model_sha256', 'posterior_profiles', 'generator_profiles_covered', 'parameters_and_forecast_validation_unchanged')}), flush=True)


if __name__ == '__main__':
    main()
