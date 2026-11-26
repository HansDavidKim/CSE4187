from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse
from scipy.special import logsumexp

from simulate.v4.irl import choice_gmm as choice, reward_gmm
from test_v4_persona_reward_gmm import actor, _source, CUTOFF, AVAILABLE


def fitted():
    latent,_=actor();actions,activity,assignments,_=_source()
    return choice.fit_choice_gmm(latent,actions,activity,assignments,
        component_model_indices={'A':0,'B':1},cutoff_at=CUTOFF,max_updates=1,qmc_power=3,
        base_model_sha256='a'*64,return_model_sha256='b'*64)


def test_zero_taste_exact_base_and_direct_temperature_choice_rule():
    latent,_=actor();solve=choice.prepare_choice_policy(latent)
    for z in range(2):
        base=np.asarray(latent['component_policies'][z]);mask=np.asarray(latent['action_mask'])
        np.testing.assert_array_equal(solve(np.zeros(4),component_id=z),base)
        theta=np.array([.4,-.2,999.,.1]);offset=np.r_[theta,0.]
        score=np.where(mask,(np.log(np.where(mask,base,1.))+offset)/2.,-np.inf)
        expected=np.where(mask,np.exp(score-logsumexp(score,axis=1,keepdims=True)),0.)
        np.testing.assert_allclose(solve(theta,component_id=z,temperature=2),expected,atol=1e-15)


def test_sparse_provider_has_zero_invalid_logs_and_return_does_not_change_choices():
    latent,_=actor();provider=choice.policy_log_provider(latent);mask=np.asarray(latent['action_mask'])
    theta=np.array([[.2,.1,-100.,.3,-2.],[.2,.1,100000.,.3,999.]])
    logs=provider(1,theta)
    assert (logs[:,~mask.ravel()]==0).all() and np.isfinite(logs).all()
    np.testing.assert_array_equal(logs[0],logs[1])
    assert np.isfinite(sparse.csr_matrix((1,mask.size))@logs.T).all()


def test_fresh_fit_keeps_known_groups_return_only_and_separate_units():
    model=fitted()
    assert model['type']==choice.TYPE and model['backend']==choice.BACKEND
    assert model['parameter_names']==choice.PARAMETERS and model['parameter_units']==choice.UNITS
    assert not model['fit']['old_reward_fit_parameters_imported']
    assert model['fit']['initial_covariance_sd']==[.5]*5
    assert all(x['training_return_only_users']==1 for x in model['fit']['personas'].values())
    choice.validate_choice_gmm(model,'a'*64,'b'*64,available_actions=AVAILABLE)
    with pytest.raises(ValueError,match='incompatible'):
        reward_gmm.validate_gmm(model,'a'*64,'b'*64)
    with pytest.raises(ValueError,match='separate choice'):
        reward_gmm.sample_reward(model,None,None,'p',np.random.default_rng(4))


def test_missing_remove_covariance_marginal_and_zero_dispersion_preserve_raw_fit():
    latent,_=actor();model=fitted();model.update(choice_temperatures=[2.,3.],dispersion_scales=[0.,1.])
    before=deepcopy(model);changed=deepcopy(model)
    for k in range(2):
        changed['means'][k][2]=999.;changed['covariances'][k][2][2]=-1.
        for j in [0,1,3,4]:changed['covariances'][k][2][j]=changed['covariances'][k][j][2]=1000.
    def draw(block):
        factors=choice.validate_choice_gmm(block,'a'*64,'b'*64,available_actions=AVAILABLE)
        return choice.sample_choice(block,factors,choice.prepare_choice_policy(latent),'p',np.random.default_rng(4),profile_components=[1.,0.])
    a,b=draw(model),draw(changed)
    assert a[0]==b[0]==0 and a[2]['return_offset']==b[2]['return_offset']==model['means'][0][4]
    np.testing.assert_array_equal(a[2]['policy'],b[2]['policy'])
    assert model==before


def test_unsupported_observation_rejected_and_outer_future_cannot_select_fit():
    latent,_=actor();actions,activity,assignments,external=_source()
    def fit(a,r):return choice.fit_choice_gmm(latent,a,r,assignments,component_model_indices={'A':0,'B':1},
        cutoff_at=CUTOFF,max_updates=1,qmc_power=3)
    expected=fit(actions,activity);changed=activity.copy()
    changed.loc[changed.user_id.eq(external)|changed.target_at.ge(CUTOFF),'label']=0
    assert fit(actions,changed)==expected
    bad=actions.copy();bad.loc[0,'action']=2
    with pytest.raises(ValueError,match='support'):fit(bad,activity)


def test_zero_dispersion_consumes_no_normal_random_draws():
    latent,_=actor();model=fitted();model['dispersion_scales']=[0.,0.]
    class Count:
        normal_calls=0
        def choice(self,*args,**kwargs):return 0
        def standard_normal(self,*args):self.normal_calls+=1;raise AssertionError('zero dispersion must not consume Gaussian RNG')
    rng=Count();factors=choice.validate_choice_gmm(model,'a'*64,'b'*64,available_actions=AVAILABLE)
    choice.sample_choice(model,factors,choice.prepare_choice_policy(latent),'p',rng,profile_components=[1.,0.])
    assert rng.normal_calls==0


def test_separate_shopping_and_return_scales_preserve_source_return_marginal():
    latent,_=actor();model=fitted();model['shopping_dispersion_scales']=[0.,0.];model['return_dispersion_scales']=[1.,1.]
    factors=choice.validate_choice_gmm(model,'a'*64,'b'*64,available_actions=AVAILABLE)
    class Fixed:
        def choice(self,*args,**kwargs):return 0
        def standard_normal(self,n):return np.arange(1,n+1,dtype=float)
    raw=deepcopy(model['covariances']);_,_,draw=choice.sample_choice(model,factors,choice.prepare_choice_policy(latent),'p',Fixed(),profile_components=[1.,0.])
    np.testing.assert_array_equal(draw['action_offsets'],np.asarray(model['means'][0])[:4])
    expected=model['means'][0][4]+(factors[0]@np.arange(1,5,dtype=float))[-1]
    assert draw['return_offset']==expected and model['covariances']==raw
    bad=deepcopy(model);bad['return_dispersion_scales']=[-1.,1.]
    with pytest.raises(ValueError,match='dispersion'):choice.validate_choice_gmm(bad,'a'*64,'b'*64)
