import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from test_v4_tianchi_daily import source,model
from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.irl import tianchi_daily as daily
from simulate.v4.simulator.tianchi_campaign import TianchiDailyCampaignEnv,model_date


@pytest.fixture
def factory(tmp_path,model):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return str(p)
    def sha(path):return hashlib.sha256(open(path,'rb').read()).hexdigest()
    def make(*,gmm=True,dispersion=0.,mutate_entry=None,mutate_gmm=None,mutate_actor=None,**kwargs):
        base=copy.deepcopy(model)
        base['source_contract']['feature_windows_days']=[28,56,91]
        base['feature_windows_days']=[28,56,91]
        base['lifecycle_memory']=daily.LIFECYCLE_MEMORY
        base['state_vectors']=np.zeros((2,len(daily.STATE_NAMES))).tolist()
        base['state_feature_means']=np.zeros((2,len(daily.STATE_NAMES))).tolist()
        base['state_phases']=[0,1]
        base['reward_weights']=np.zeros((2,4,len(daily.REWARD_FEATURE_NAMES))).tolist()
        base['mask']=[[True,True,True,True,False]]*2
        stay={'shape':[2,2],'data':[1.,1.],'indices':[0,1],'indptr':[0,1,2]}
        home={'shape':[2,2],'data':[1.,1.],'indices':[1,1],'indptr':[0,1,2]}
        empty={'shape':[2,2],'data':[],'indices':[],'indptr':[0,0,0]}
        base['transition_csr']=[stay,home,home,home,empty]
        paths={}
        for group in ('0','1'):
            actor=copy.deepcopy(base);actor['training']['persona_cluster']=group
            if mutate_actor:mutate_actor(actor)
            paths[group]=write(f'actor{group}.json',actor)
        verification=write('verification.json',{'actor_models':{k:sha(p) for k,p in paths.items()}})
        entry=dict(schema_version=1,type='tianchi_daily_initial_entry_priors',
            base_model_sha256=sha(verification),daily_actor_model_sha256={k:sha(p) for k,p in paths.items()},
            source_contract=base['source_contract'],observation_contract=base['observation_contract'],
            persona_labels=['0','1'],persona_probabilities=[.6,.4],entry_outcome_order=list(daily.OUTCOMES[1:]),
            entry_outcome_probabilities={'0':[1.,0.,0.],'1':[1.,0.,0.]},
            training={'cutoff_at':base['training']['cutoff_at'],'max_entry_known_at':'2016-01-02T00:00:00+00:00',
                      'training_users_only':True,'first_active_day_is_conditioned':True})
        if mutate_entry:mutate_entry(entry)
        settings={'backend':'tianchi_daily_campaign','daily_actor_model_paths':paths,
            'daily_actor_verification_path':verification,'initial_entry_model_path':write('entry.json',entry)}
        if gmm:
            block={**{k:entry[k] for k in ('base_model_sha256','daily_actor_model_sha256','source_contract','observation_contract','persona_labels')},
                'schema_version':1,'type':'source_persona_daily_reward_gmm','backend':'tianchi_daily','purpose':'synthesis_only',
                'parameter_names':list(daily.TRAIT_NAMES),'component_count':2,'component_model_indices':[0,1],
                'condition_on_profile_components':True,'available_actions':['view','purchase'],
                'fitted_parameter_names':['view','purchase','return'],'mixture_weights':[.6,.4],
                'means':[[.1,10000,-50000,.2,-.05]]*2,'covariances':np.repeat(np.eye(5)[None],2,axis=0).tolist(),
                'choice_temperatures':[1.,.75],'dispersion_scales':[dispersion]*2}
            if mutate_gmm:mutate_gmm(block)
            settings['reward_gmm_model_path']=write('gmm.json',block)
        config={'simulation':{'start_at':'2024-01-01T00:00:00+09:00'},'source_simulation':settings}
        return TianchiDailyCampaignEnv(config,warmup_days=kwargs.pop('warmup_days',30),
            min_history_days=0,horizon_days=kwargs.pop('horizon_days',14),**kwargs)
    return make


def oracle(env):
    history=pd.DataFrame([dict(user_id=env.uid,day_start=r['model_day'],view=r['view'],purchase=r['purchase']) for r in env.records['raw_daily']])
    expected=daily.public_observation(history,model_date(env.now),source_contract=env.source_contract)
    values=np.asarray([np.nan if expected[k] is None else expected[k] for k in EXPANDED_FEATURES])
    np.testing.assert_allclose(env._features(),values,equal_nan=True)


def provider(outcome):
    def callback(actor,theta):
        logs=np.full((len(theta),2,5),-np.inf)
        logs[:,:,outcome]=0.;logs[:,:,4]=0.
        return logs.reshape(len(theta),-1)
    return callback


def test_date_coordinate_mapping_requires_real_business_midnight():
    assert model_date('2024-01-01T00:00:00+09:00')==pd.Timestamp('2024-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='midnights'):model_date('2024-01-01T00:00:00Z')
    with pytest.raises(ValueError,match='timezone-aware'):model_date('2024-01-01')


def test_whole_weekly_daily_slices_preserve_same_theta_history_and_nullable39(factory):
    whole=factory(horizon_days=35);whole.reset(100)
    weekly=copy.copy(whole);sliced=copy.copy(whole)
    theta=whole.theta.copy()
    whole.advance(0,35)
    for _ in range(5):weekly.advance(0,7);oracle(weekly)
    for _ in range(35):sliced.advance(0,1);oracle(sliced)
    for name in ('raw_daily','daily_diagnostics','runtime_lifecycle'):
        pd.testing.assert_frame_equal(whole.tables[name],weekly.tables[name])
        pd.testing.assert_frame_equal(whole.tables[name],sliced.tables[name])
    np.testing.assert_array_equal(whole.theta,theta)
    assert whole.theta[1]==whole.theta[2]==0
    assert len(whole.records['raw_daily'])==65
    unsupported=[EXPANDED_FEATURES.index(k) for k in EXPANDED_FEATURES if k not in daily.PUBLIC_ROLES]
    assert np.isnan(whole._features()[unsupported]).all()
    assert 'sessions' not in whole.records and 'events' not in whole.records and 'orders' not in whole.records
    assert whole.tables['raw_daily'].known_at.le(whole.now).all()


def test_pending_day_never_enters_observation_or_changes_after_future_coupon(factory):
    shifts=lambda env,at:{'purchase_reward_shift':5. if env._receipts else 0.}
    response=lambda env,at:{'response_selected':True,'response_ready_at':at}
    env=factory(warmup_days=1,response_sampler=response,runtime_reward_offsets=shifts)
    def callback(actor,theta):
        return provider(2 if theta[0,3]>1 else 0)(actor,theta)
    env._policy_provider=callback
    env.reset(101)
    before=env._features().copy();count=len(env.records['raw_daily'])
    env._plan_day();pending=copy.deepcopy(env._pending)
    assert pending['outcome']=='no_activity'
    np.testing.assert_allclose(env._features(),before,equal_nan=True)
    assert len(env.records['raw_daily'])==count
    clone=copy.copy(env)
    assert clone._daily.rng is clone.day_rng and clone.day_rng is not env.day_rng
    assert clone._pending is not env._pending
    values,reward,done,info=clone.advance(1,1)
    assert clone.records['raw_daily'][-1]['outcome']=='no_activity'
    assert reward==-100 and info['converted_sessions']==0
    assert env.records['raw_daily']==[env.records['raw_daily'][0]] and not env._receipts
    _,reward,_,info=clone.advance(0,1)
    assert info['converted_purchase_days']==info['converted_sessions']==info['redemptions']==1 and reward==1000
    assert info['paid_minor'] is info['discount_minor'] is info['orders'] is None
    oracle(clone)


def test_private_waiting_exit_uses28_known_absent_dates_and_reentry_uses_past_active_count(factory):
    shifts=lambda env,at:{'return_reward_shift':20. if env._receipts else 0.}
    response=lambda env,at:{'response_selected':False,'response_ready_at':None}
    env=factory(warmup_days=1,horizon_days=40,response_sampler=response,runtime_reward_offsets=shifts)
    env._policy_provider=lambda actor,theta:provider(1 if theta[0,4]>1 else 0)(actor,theta)
    env.reset(102);theta=env.theta.copy()
    env.advance(0,27)
    assert env._lifecycle_phase=='CHURNING' and env._inactive_days==27
    env.advance(0,1)
    assert env._lifecycle_phase=='EXIT' and env._inactive_days==28
    env.advance(1,1)
    assert env.records['daily_diagnostics'][-1]['actor_phase_before']=='NEW'
    env.advance(0,1)
    assert env.records['daily_diagnostics'][-1]['actor_phase_before']=='HOME'
    assert env._inactive_days==0
    np.testing.assert_array_equal(env.theta,theta)
    assert any(r['reason']=='known_active_date_reentry' for r in env.records['runtime_lifecycle'])


def test_zero_dispersion_draw_uses_no_noise_and_unsupported_covariance_is_ignored(factory):
    a=factory(warmup_days=1);a.reset(103)
    streams=np.random.SeedSequence([103,0]).spawn(4)
    expected=np.random.default_rng(streams[0]);expected.choice(2,p=[.6,.4])
    assert a.rng.bit_generator.state==expected.bit_generator.state
    def nuisance(block):
        for cov in block['covariances']:
            for j in (1,2):
                cov[j]=[9999.]*5
                for i in range(5):cov[i][j]=-12345.
    b=factory(warmup_days=1,mutate_gmm=nuisance);b.reset(103)
    a.advance(0,14);b.advance(0,14)
    np.testing.assert_array_equal(a.theta,b.theta)
    pd.testing.assert_frame_equal(a.tables['raw_daily'],b.tables['raw_daily'])


def test_zero_effect_contacts_preserve_daily_rng_and_S_D_only_change_paper_cost(factory):
    response=lambda env,at:{'response_selected':False,'response_ready_at':None}
    no=factory(warmup_days=1);s=factory(warmup_days=1,cost_model='S',response_sampler=response)
    d=factory(warmup_days=1,cost_model='D',response_sampler=response)
    for env in (no,s,d):env.reset(104)
    _,base,_,_=no.advance(0)
    _,score_s,_,info_s=s.advance(1)
    _,score_d,_,info_d=d.advance(1)
    assert score_s==base-100 and score_d==base
    assert info_s['send_cost_minor']==info_d['send_cost_minor']==100
    for env in (s,d):
        pd.testing.assert_frame_equal(no.tables['raw_daily'],env.tables['raw_daily'])
        assert no.day_rng.bit_generator.state==env.day_rng.bit_generator.state


@pytest.mark.parametrize('cost_model',['S','D'])
def test_objective_cost_changes_reward_without_changing_daily_paths(factory,cost_model):
    response=lambda env,at:{'response_selected':True,'response_ready_at':at}
    old=factory(warmup_days=1,cost_model=cost_model,response_sampler=response)
    high=factory(warmup_days=1,cost_model=cost_model,response_sampler=response,score_coupon_cost=500.)
    for env in (old,high):env._policy_provider=provider(2)
    np.testing.assert_allclose(old.reset(141),high.reset(141),equal_nan=True)
    purchases=uses=0
    for action in (1,1):
        before,reward,done,info=old.advance(action)
        after,new_reward,new_done,new_info=high.advance(action)
        np.testing.assert_allclose(before,after,equal_nan=True)
        assert done==new_done
        units=info['sends' if cost_model=='S' else 'redemptions']
        assert new_reward==reward-400*units
        assert new_info['score_cost']==500*units
        assert new_info['score_conversion_value']==1000 and new_info['score_coupon_cost']==500
        assert new_info['send_cost_minor']==info['send_cost_minor']==100
        assert new_info['orders'] is new_info['paid_minor'] is new_info['discount_minor'] is None
        purchases+=info['converted_sessions'];uses+=info['redemptions']
    assert purchases>0 and uses>0
    for name in ('raw_daily','coupons','coupon_redemptions','runtime_lifecycle','daily_diagnostics'):
        pd.testing.assert_frame_equal(old.tables[name],high.tables[name])
    assert old.day_rng.bit_generator.state==high.day_rng.bit_generator.state
    assert high.records['_metadata']['score_coupon_cost']==500
    assert high.records['_metadata']['cost_model']==cost_model


@pytest.mark.parametrize('name,value',[
    ('score_conversion_value',0),('score_conversion_value',True),('score_conversion_value',np.inf),
    ('score_conversion_value','1000'),('score_coupon_cost',-1),('score_coupon_cost',np.bool_(True)),
    ('score_coupon_cost',np.nan),('score_coupon_cost','500'),
])
def test_daily_objective_rejects_invalid_weights(factory,name,value):
    with pytest.raises(ValueError,match=name):factory(**{name:value})


@pytest.mark.parametrize('change',[
    lambda b:b.update(base_model_sha256='bad'),
    lambda b:b['daily_actor_model_sha256'].update({'0':'bad'}),
    lambda b:b['source_contract'].update(prepared_sha256='different'),
    lambda b:b.update(backend='retail_session'),
    lambda b:b.update(choice_temperatures=[0.,1.]),
])
def test_daily_gmm_binding_rejects_foreign_backends_or_stale_assets(factory,change):
    with pytest.raises(ValueError):factory(mutate_gmm=change)


def test_future_entry_and_unconfigured_receipts_are_explicitly_rejected(factory):
    with pytest.raises(ValueError,match='mature'):
        factory(mutate_entry=lambda e:e['training'].update(max_entry_known_at=e['training']['cutoff_at']))
    env=factory(warmup_days=1);env.reset(105)
    with pytest.raises(ValueError,match='response_sampler'):env.advance(1)
    assert not env.records['coupons']


def test_weekly_midnight_exposes_completed_sunday_and_keeps_monday_pending_private(factory):
    env=factory(warmup_days=1);env.reset(106);env.advance(0,6)
    assert env.now.tz_convert('Asia/Seoul')==pd.Timestamp('2024-01-08T00:00:00+09:00')
    last=env.records['raw_daily'][-1]
    assert last['known_at']==env.now
    assert last['day_end']-last['day_start']==daily.DAY
    assert last['model_day']==model_date(env.now)-daily.DAY
    assert last['model_day']-last['day_start']==pd.Timedelta(hours=9)
    before=env._features().copy();count=len(env.records['raw_daily'])
    env._plan_day()
    assert env._pending['model_day']==model_date(env.now) and env._pending['known_at']==env.now+daily.DAY
    assert len(env.records['raw_daily'])==count
    np.testing.assert_allclose(before,env._features(),equal_nan=True)


def test_temperature_scales_base_and_canonical_incidence_together_through_bellman(factory):
    def mutate(model):
        weights=np.asarray(model['reward_weights'])
        weights[0,:,0]=[.3,-.1,.4,-.3]
        weights[1,:,0]=[-.2,.5,-.1,.2]
        model['reward_weights']=weights.tolist()
    env=factory(warmup_days=1,mutate_actor=mutate,mutate_gmm=lambda b:b.update(choice_temperatures=[.5,.5]))
    env.reset(107);env._plan_day()
    model=env.models[env.actor]
    _,_,_,matrices,_,base=daily._model_arrays(model)
    reward=np.column_stack([(base+daily.TRAIT_INCIDENCE@env.theta)/.5,np.zeros(len(base))])
    expected=daily.shopping.soft_policy(reward,matrices,np.asarray(model['mask']),model['discount'])[0][:,:4]
    np.testing.assert_allclose(env._policy_cache[(0.,0.)],expected,atol=1e-9,rtol=1e-9)


def test_weekly_constructor_rejects_legacy30day_basis_and_reset_clears_private_scenario_state(factory):
    with pytest.raises(ValueError,match='4/8/13w'):
        factory(mutate_actor=lambda model:model['source_contract'].pop('feature_windows_days'))
    env=factory(warmup_days=1);env.reset(108)
    env.state['coupon_response']={'private_scenario':'old customer'}
    clone=copy.copy(env)
    clone.state['coupon_response']['private_scenario']='clone only'
    assert env.state['coupon_response']['private_scenario']=='old customer'
    env.reset(109)
    assert 'coupon_response' not in env.state and env.state['observed_active_days']==1
