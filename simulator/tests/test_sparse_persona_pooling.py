from copy import deepcopy

import numpy as np
from scipy import sparse

from scripts.build_sparse_persona_pooling import shared_source_prior
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl.linear_actor import FEATURES,validate_linear_actor
from simulate.v4.irl import choice_gmm as choice


def actor():
    encoder={'feature_names':shopping.FEATURE_NAMES,'state_keys':[[0,0,0],[0,0,1]],
        'context_means':np.zeros((2,40)).tolist(),'feature_scale':np.ones(40).tolist(),
        'tree':{'left':[-1],'right':[-1],'feature':[-1],'threshold':[0.]},
        'available_actions':['view','cart','purchase'],'lifecycle_memory':'first_two_observed_sessions'}
    mask=shopping.action_mask(encoder);basis=np.zeros((2,80));basis[:,0]=1
    phases=np.array([0,1]);theta=np.zeros((3,2,4,80))
    theta[0,:,0,0]=[.3,.4];theta[1,:,3,0]=[.1,.2];theta[2,:,0,0]=[3.,4.]
    rewards=np.stack([np.einsum('nf,naf->na',basis,theta[k,phases]) for k in range(3)])
    weights=np.array([.1,.89,.01]);common=np.einsum('k,knj->nj',weights,rewards)
    matrices=[sparse.eye(2,format='csr') if a in (0,1,3) else sparse.csr_matrix((2,2)) for a in range(5)]
    solutions=[shopping.soft_policy(np.column_stack([reward,np.zeros(2)]),matrices,mask) for reward in rewards]
    block={'schema_version':1,'type':shopping.TYPE,'actions':list(shopping.ACTIONS),'discount':shopping.DISCOUNT,
        'regularization':.01,'state_count':2,'feature_names':shopping.FEATURE_NAMES,'context_keys':shopping.CONTEXT_KEYS,
        'encoder':encoder,'action_mask':mask.tolist(),'transition_csr':[shopping.pack_matrix(p) for p in matrices],
        'mixture_weights':weights.tolist(),'component_count':3,'persona_labels':['0','1','2'],
        'common_reward':common.tolist(),'component_reward_deviations':(rewards-common).tolist(),
        'component_offsets':np.zeros((3,4)).tolist(),'component_policies':[p[0].tolist() for p in solutions],
        'component_values':[p[1].tolist() for p in solutions],'profile_posteriors':{'a':[1.,0.,0.]},
        'linear_reward':{'feature_names':list(FEATURES),'state_basis':basis.tolist(),'state_phase_ids':phases.tolist(),
            'weights':theta.tolist()},'training':{'persona_models':{
                '0':{'action_users':1205,'action_records':9955},'1':{'action_users':12240,'action_records':35750},
                '2':{'action_users':8,'action_records':857}}},'return_activity':{'preserved':[.2,-.1,.7]},'limitations':[]}
    shopping.validate_model(block)
    return block


def test_sparse_support_uses_distinct_source_users_and_keeps_other_personas_exact():
    original=actor();before=deepcopy(original)
    changed,report=shared_source_prior(original,minimum_independent_users=50)
    assert original==before and report['selected_personas']==['2'] and not report['optimization_performed']
    assert report['support_users']=={'0':1205,'1':12240,'2':8}
    assert changed['return_activity']==original['return_activity']
    for key in ('encoder','transition_csr','action_mask','common_reward','mixture_weights','component_offsets'):
        assert changed[key]==original[key]
    for key in ('component_reward_deviations','component_policies','component_values'):
        assert changed[key][:2]==original[key][:2]
    assert changed['linear_reward']['weights'][:2]==original['linear_reward']['weights'][:2]
    shared=np.einsum('k,kpaf->paf',original['mixture_weights'],original['linear_reward']['weights'])
    np.testing.assert_array_equal(changed['linear_reward']['weights'][2],shared)
    assert not np.asarray(changed['component_reward_deviations'][2]).any()
    shopping.validate_model(changed);validate_linear_actor(changed)
    assert np.asarray(changed['component_policies'][2])[:,4].min()>np.asarray(original['component_policies'][2])[:,4].max()


def test_explicit_support_boundary_does_not_pool_a_group_at_the_minimum():
    original=actor();original['training']['persona_models']['2']['action_users']=50
    changed,report=shared_source_prior(original,minimum_independent_users=50)
    assert report['selected_personas']==[]
    assert changed['component_policies']==original['component_policies']


def test_zero_sparse_shopping_scale_preserves_return_draw_and_full_rng_order():
    original=actor();changed,_=shared_source_prior(original)
    block={'schema_version':1,'type':choice.TYPE,'backend':choice.BACKEND,'purpose':'synthesis_only',
        'parameter_names':choice.PARAMETERS,'parameter_units':choice.UNITS,'component_count':3,
        'persona_labels':['0','1','2'],'component_model_indices':[0,1,2],
        'condition_on_profile_components':True,'available_actions':['view','cart','purchase'],
        'fitted_parameter_names':[choice.PARAMETERS[i] for i in (0,1,3,4)],
        'mixture_weights':[.1,.89,.01],'means':np.zeros((3,5)).tolist(),
        'covariances':np.repeat(np.eye(5)[None]*.25,3,axis=0).tolist(),
        'base_model_sha256':'a'*64,'return_model_sha256':'b'*64,
        'shopping_dispersion_scales':[0.,0.,1.],'return_dispersion_scales':[1.,1.,1.]}
    modified=deepcopy(block);modified['shopping_dispersion_scales'][2]=0.
    factors=choice.validate_choice_gmm(block,'a'*64,'b'*64)
    left=np.random.default_rng(19);right=np.random.default_rng(19)
    old=choice.sample_choice(block,factors,choice.prepare_choice_policy(original),'p',left,profile_components=[0.,0.,1.])
    new=choice.sample_choice(modified,factors,choice.prepare_choice_policy(changed),'p',right,profile_components=[0.,0.,1.])
    assert old[0]==new[0]==2 and old[2]['return_offset']==new[2]['return_offset']
    assert left.bit_generator.state==right.bit_generator.state
    assert not new[2]['action_offsets'].any()
    assert block['means']==modified['means'] and block['covariances']==modified['covariances']
