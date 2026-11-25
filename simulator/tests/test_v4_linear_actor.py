from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import linear_actor as linear
from simulate.v4.irl import fit_latent_shopping as shopping
from simulate.v4.irl import fit_joint_latent_shopping as joint
from simulate.v4.irl.source_observed_features import build_source_observations
from simulate.v4.time_spec import weekly_metadata


def test_existing79_transform_uses_training_only_and_adds_no_presence_columns():
    core=np.zeros((4,40));core[:,3]=[0,1,2,3]
    raw=np.full((4,39),np.nan);raw[:,2]=[1,2,3,4]
    frozen=linear.fit_transform(core[:3],raw[:3])
    x=linear.transformed(core,raw,frozen)
    assert x.shape==(4,80) and len(linear.FEATURES)==80
    assert np.isfinite(x).all() and np.all(x[:,1:3]==0)
    before=x[:3].copy();raw[3,2]=1e12
    np.testing.assert_array_equal(before,linear.transformed(core,raw,frozen)[:3])
    assert not any('present:' in name for name in linear.FEATURES)


def test_compiled_linear_rewards_bind_every_persona_and_private_phase():
    basis=np.zeros((4,80));basis[:,0]=1;basis[:,1]=[0,1,2,3]
    phases=np.array([0,0,1,1]);theta=np.zeros((2,2,4,80))
    theta[0,:,0,1]=[1,2];theta[1,:,3,0]=[3,4]
    rewards=np.stack([np.einsum('np,nap->na',basis,theta[k,phases]) for k in range(2)])
    common=rewards.mean(axis=0)
    block={'mixture_weights':[.5,.5],'state_count':4,'common_reward':common,
           'component_reward_deviations':rewards-common,
           'linear_reward':{'feature_names':list(linear.FEATURES),'state_basis':basis,
                            'state_phase_ids':phases,'weights':theta}}
    linear.validate_linear_actor(block)
    broken=deepcopy(block);broken['linear_reward']['weights'][1,1,3,0]+=1
    with pytest.raises(ValueError,match='differs'):
        linear.validate_linear_actor(broken)


def toy_problem(monkeypatch):
    cutoff=pd.Timestamp('2020-01-29',tz='UTC');origin=pd.Timestamp('2019-10-01',tz='UTC')
    rows=[];sid=0
    for uid in range(32):
        days=[0,1,5,10,35,65,100,145] if uid%2==0 else [0,30,80,145]
        for day in days:
            sid+=1
            actions=['view','cart','purchase'] if uid%2==0 else ['view','cart','remove']
            for step,action in enumerate(actions):
                rows.append(dict(user_id=str(uid),session=sid,at=origin+pd.Timedelta(days=day,seconds=step),
                    action=action,event_type='remove_from_cart' if action=='remove' else action,
                    product_id='a',price_minor=100,source_event_id=f'{sid:06d}:{step}'))
    rows=pd.DataFrame(rows).sort_values(['user_id','at','source_event_id'],kind='stable')
    end=origin+pd.Timedelta(days=150)
    data=shopping.build_source(rows,end,lifecycle_memory='first_two_observed_sessions')
    data['source_rows']=rows
    metadata={**weekly_metadata(),'observation_start_at':origin,'observation_end_at':end}
    raw,_=build_source_observations(rows,data['records'],metadata=metadata)
    assignments=pd.DataFrame({'user_id':[str(i) for i in range(32)],'cluster':[i%2 for i in range(32)],'assignment_cutoff':cutoff})
    contract={'domain':'cosmetics','observed_actions':['view','cart','remove','purchase'],
              'training_cutoff_at':cutoff.isoformat(),'outer_cutoff_at':cutoff.isoformat(),
              'inner_cutoff_at':'2019-12-29T00:00:00+00:00'}
    base={'source_contract':contract,'seed':42,'observation_end':end.isoformat()}
    returning={'validation':{'temporal_cutoff':cutoff.isoformat()}}
    monkeypatch.setattr(joint,'baseline_logit',lambda model,context:0.)
    return data,raw,assignments,base,returning


def test_actual_toy_persona_phase_fit_compiles_main_policy_without_new_covariates(monkeypatch):
    data,raw,assignments,base,returning=toy_problem(monkeypatch)
    cutoff=pd.Timestamp(base['source_contract']['training_cutoff_at']);end=pd.Timestamp(base['observation_end'])
    actor=linear.fit_grouped_actor(data,raw,assignments,base,returning,'a'*64,
                                   regularization=.1,max_iter=300)
    shopping.validate_model(actor);joint.validate_model(actor,'a'*64);linear.validate_linear_actor(actor)
    assert actor['persona_labels']==['0','1']
    assert np.asarray(actor['linear_reward']['weights']).shape==(2,2,4,80)
    assert actor['feature_names']==shopping.FEATURE_NAMES and len(actor['feature_names'])==40
    assert len(actor['linear_reward']['feature_names'])==80
    assert not np.allclose(actor['component_policies'][0],actor['component_policies'][1])
    assert sum(actor['mixture_weights'])==pytest.approx(1)
    scores=linear.score_future_actions(actor,data,assignments,start_at=cutoff,end_at=end)
    assert sum(value['records'] for value in scores.values())>0
    assert all(value['customer_macro_nll'] is None or np.isfinite(value['customer_macro_nll']) for value in scores.values())
    unavailable=assignments.copy()
    unavailable['assignment_cutoff']=end
    with pytest.raises(ValueError,match='unavailable persona'):
        linear.score_future_actions(actor,data,unavailable,start_at=cutoff,end_at=end)
    unknown=linear.score_future_actions(actor,data,assignments.iloc[:0],start_at=cutoff,end_at=end)
    assert unknown['__pooled__']['records']==scores['__pooled__']['records']
    assert unknown['__pooled__']['unknown_prefix_users']==unknown['__pooled__']['users']
    training=[str(uid) for uid in range(32) if hashlib.sha256(f'42:{uid}'.encode()).digest()[0]<205]
    sparse=assignments.copy()
    sparse['cluster']=sparse.user_id.map({training[0]:'0',training[1]:'1'}).fillna('2')
    fallback=linear.fit_grouped_actor(data,raw,sparse,base,returning,'a'*64,
        selected_penalties={'0':.01,'1':10.,'2':.1},max_iter=300)
    assert fallback['training']['persona_models']['0']['pooled_fallback']
    assert fallback['training']['persona_models']['1']['pooled_fallback']
    assert not np.allclose(fallback['linear_reward']['weights'][0],fallback['linear_reward']['weights'][1])


@pytest.mark.parametrize('penalties',[None,{'0':.1,'1':.2}])
def test_optional_pooled_prior_uses_actual_same_penalty_training_coefficients(monkeypatch,penalties):
    data,raw,assignments,base,returning=toy_problem(monkeypatch)
    calls=[];original=linear.fit_feature_reward
    def record(*args,**kwargs):
        result=original(*args,**kwargs)
        calls.append((kwargs,result))
        return result
    monkeypatch.setattr(linear,'fit_feature_reward',record)
    actor=linear.fit_grouped_actor(data,raw,assignments,base,returning,'a'*64,
        regularization=.1,selected_penalties=penalties,max_iter=300,prior_mode='pooled_feature_reward')
    shopping.validate_model(actor);joint.validate_model(actor,'a'*64)
    pool={kw['regularization']:fit for kw,fit in calls if 'prior' not in kw}
    groups=[(kw,fit) for kw,fit in calls if 'prior' in kw]
    assert len(pool)==(1 if penalties is None else 2) and len(groups)==2
    for kwargs,fitted in groups:
        center=pool[kwargs['regularization']]['reward_weights']
        np.testing.assert_array_equal(kwargs['prior'],center)
        np.testing.assert_array_equal(kwargs['initial'],center)
        assert fitted['fit']['converged'] and 'theta-prior' in fitted['fit']['objective_definition']
    metadata=actor['linear_reward']['coefficient_prior']
    assert metadata['mode']=='pooled_feature_reward'
    assert set(metadata['pooled_weights_by_penalty'])==set(map(str,pool))
    assert len(actor['linear_reward']['feature_names'])==80 and len(actor['feature_names'])==40
    with pytest.raises(ValueError,match='prior_mode'):
        linear.fit_grouped_actor(data,raw,assignments,base,returning,'a'*64,prior_mode='unknown')
