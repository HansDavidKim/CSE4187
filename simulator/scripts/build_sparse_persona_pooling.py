"""Compile an opt-in source-weighted linear IRL prior for sparse personas.

No likelihood is optimized here. Existing source mixture weights and linear
coefficients define the shared prior; the original dynamics solve its policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np

from simulate.v4.configuration import load_config
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl.linear_actor import validate_linear_actor
from simulate.v4.irl.choice_gmm import validate_choice_gmm
from simulate.v4.irl.source_contract import file_sha


def shared_source_prior(model, *, minimum_independent_users=50):
    """Return a new actor; group support is distinct training users, not rows."""
    if type(minimum_independent_users) is not int or minimum_independent_users<1:
        raise ValueError('minimum independent users must be a positive integer')
    shopping.validate_model(model)
    result=deepcopy(model);block=result.get('latent_shopping',result)
    linear=block.get('linear_reward')
    if linear is None or np.asarray(block['component_offsets']).any():
        raise ValueError('shared linear prior requires compiled linear rewards without separate action offsets')
    labels=block['persona_labels'];reports=block['training']['persona_models']
    users=[reports[label]['action_users'] for label in labels]
    if any(type(n) is not int or n<0 for n in users):raise ValueError('independent source training user counts are required')
    selected=[index for index,n in enumerate(users) if n<minimum_independent_users]
    weights=np.asarray(block['mixture_weights'],float)
    theta=np.asarray(linear['weights'],float)
    shared=np.einsum('k,kpaf->paf',weights,theta)
    basis=np.asarray(linear['state_basis'],float);phases=np.asarray(linear['state_phase_ids'],int)
    common=np.asarray(block['common_reward'],float)
    expected=np.einsum('nf,naf->na',basis,shared[phases])
    if not np.allclose(expected,common,rtol=1e-10,atol=1e-10):
        raise ValueError('existing common reward is not the source-weighted linear coefficient prior')
    probability,value,residual=shopping.soft_policy(np.column_stack([common,np.zeros(len(common))]),
        shopping.unpack_matrices(block),np.asarray(block['action_mask'],bool),block['discount'])
    for index in selected:
        linear['weights'][index]=shared.tolist()
        block['component_reward_deviations'][index]=np.zeros_like(common).tolist()
        block['component_policies'][index]=probability.tolist();block['component_values'][index]=value.tolist()
    report={'type':'source_weighted_shared_linear_irl_prior','minimum_independent_training_users':minimum_independent_users,
        'support_statistic':'distinct action-training source users; never action rows',
        'support_users':dict(zip(labels,users)),'selected_personas':[labels[index] for index in selected],
        'source_mixture_weights':weights.tolist(),'policy_bellman_residual':residual,
        'common_reward_anchor':'original source-mixture-weighted coefficient reward, frozen before replacement',
        'weights_rule':'sum(original source mixture weight * original persona-phase linear coefficients)',
        'optimization_performed':False,'interpretation':'shared source-weighted IRL prior, not independently optimized pooled MCE',
        'decision_timing':'adaptive repair after cap diagnosis; source-only support rule registered before new-world tests',
        'unchanged':'encoder,dynamics,mask,39+40/intercept basis,source mixture,other personas,return model/offsets',
        'cap_rule':'original RuntimeError unchanged; no END floor, cap increase, drop or reseed'}
    block['sparse_persona_pooling']=report
    block['limitations'].append('Sparse-persona shared prior is an adaptive post-diagnostic repair; old-world validation is not new-world evidence.')
    validate_linear_actor(block);shopping.validate_model(result)
    return result,report


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def build_assets(config_path, output, *, minimum_independent_users=50):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    config=load_config(config_path);settings=config['source_simulation']
    model_path=Path(settings['model_path']);gmm_path=Path(settings['choice_gmm_model_path'])
    canonical_path=Path(settings['canonical_response_model_path'])
    originals={model_path,gmm_path,canonical_path,Path(config_path).resolve(),
        *(Path(v) for k,v in settings.items() if k.endswith('_path') and isinstance(v,str))}
    old_model=json.loads(model_path.read_text());old_gmm=json.loads(gmm_path.read_text())
    model,report=shared_source_prior(old_model,minimum_independent_users=minimum_independent_users)
    actor=model['latent_shopping'];selected=[actor['persona_labels'].index(label) for label in report['selected_personas']]
    if not selected:raise ValueError('no sparse persona meets the explicit source support criterion')
    gmm=deepcopy(old_gmm);count=gmm['component_count']
    shopping_scales=list(gmm.get('shopping_dispersion_scales',gmm.get('dispersion_scales',[1.]*count)))
    for group,actor_index in enumerate(gmm['component_model_indices']):
        if actor_index in selected:
            if np.asarray(gmm['means'][group])[:4].any():raise ValueError('sparse shopping means are not the zero shared-policy reference')
            shopping_scales[group]=0.
    gmm['shopping_dispersion_scales']=shopping_scales
    gmm['raw_fit_base_model_sha256']=old_gmm['base_model_sha256']
    gmm['runtime_sparse_persona_pooling']={**report,'raw_means_covariances_return_margins_preserved':True,
        'draw_order':'same component choice and full observed-coordinate Gaussian draw; return scale unchanged'}
    output.mkdir(parents=True)
    write(output/'latent_shopping.json',actor)
    write(output/'shopping_joint_latent_source_traits.json',model)
    gmm['base_model_sha256']=file_sha(output/'shopping_joint_latent_source_traits.json')
    validate_choice_gmm(gmm,gmm['base_model_sha256'],file_sha(settings['return_model_path']),
        available_actions=actor['encoder']['available_actions'])
    write(output/'source_choice_gmm.json',gmm)
    bare=deepcopy(config);bare['source_simulation'].update(model_path=str(output/'shopping_joint_latent_source_traits.json'),
        choice_gmm_model_path=str(output/'source_choice_gmm.json'))
    bare['source_simulation'].pop('canonical_response_model_path')
    write(output/'source_campaign.json',bare)
    canonical=json.loads(canonical_path.read_text());old_calibration=canonical.pop('fresh_world_calibration',None)
    canonical['previous_world_calibration']=old_calibration
    canonical['new_world_calibrated']=False
    canonical['strength_status']='previous-world selected starting values only; new shared-prior world requires fresh calibration'
    canonical['previous_world_behavior_sha256']=deepcopy(canonical['behavior_sha256'])
    for key in ('model_path','choice_gmm_model_path'):
        canonical['behavior_sha256'][key]=file_sha(bare['source_simulation'][key])
    template=output/'canonical_template';template.mkdir()
    bank=canonical_path.parent/canonical['latent_bank_file'];originals.add(bank.resolve())
    shutil.copyfile(bank,template/'latent_bank.parquet')
    canonical['latent_bank_file']='latent_bank.parquet'
    canonical['latent_bank_sha256']=file_sha(template/'latent_bank.parquet')
    write(template/'model.json',canonical)
    linked=deepcopy(bare);linked['source_simulation']['canonical_response_model_path']=str(template/'model.json')
    write(output/'canonical_config.json',linked)
    report.update(status='new_assets_compiled_no_new_fit',
        original_input_sha256={str(path):file_sha(path) for path in sorted(originals)},
        new_asset_sha256={str(path):file_sha(path) for path in sorted(output.rglob('*')) if path.is_file()},
        canonical_new_world_calibrated=False,source_config=str(output/'source_campaign.json'),canonical_config=str(output/'canonical_config.json'))
    write(output/'build_report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--minimum-independent-users',type=int,default=50)
    args=parser.parse_args()
    result=build_assets(args.config,args.output,minimum_independent_users=args.minimum_independent_users)
    print(json.dumps({key:result[key] for key in ('status','support_users','selected_personas','source_config','canonical_config')}),flush=True)
