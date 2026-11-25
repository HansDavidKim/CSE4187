"""Aggregate Main profiles and opt-in coupon logit shifts preserve the IRL baseline."""
import copy

import numpy as np
import pandas as pd
import pytest

from simulate.v4.simulator import main_campaign as main
from simulate.v4.simulator import population_profiles, population_timing
from test_v4_main_campaign import factory


def population_edit(model):
    model.pop('profiles'); model.pop('catalog')
    model['latent_shopping']['profile_posteriors']={}
    model['source']={'sample_sha256':None}
    group={'customers':10,'component_probabilities':[.25,.75], 'visit_probability_30d':.8,
        'preferred_price_counts':[[1000,1]],'assortment_size_counts':[[1,1]],'leading_preference_share':1.,
        'checkout_preference_share':0.,'choice_gap_counts':[[0,1]]}
    model['population']={'schema_version':1,'catalog_bands':[[[1000,1,3]]],
        'home_groups':{'buyer':group},'checkout_counts':[[2,1,1000,5]]}


def test_compiled_catalog_keeps_all_draws_and_reuses_only_deterministic_bounds(factory,monkeypatch):
    monkeypatch.setattr(population_timing,'materialize_visit_timing',lambda value,source:value)
    env=factory(edit=population_edit,horizon_days=14)
    index=population_profiles.compile_population_catalog(env.product_rows)
    for seed in (17,29,42):
        a,b=np.random.default_rng(seed),np.random.default_rng(seed)
        left,draw_left=population_profiles.prepare_population(env.population,env.product_rows,a)
        right,draw_right=population_profiles.prepare_population(env.population,env.product_rows,b,catalog_index=index)
        assert left==right
        assert [draw_left(n) for n in range(5)]==[draw_right(n) for n in range(5)]
        assert a.bit_generator.state==b.bit_generator.state
    assert index[2] and not index[0].flags.writeable and not index[1].flags.writeable
    env.reset(7)
    cloned=copy.deepcopy(env)
    assert cloned.population_catalog_index is env.population_catalog_index
    assert cloned.home_groups is not env.home_groups


def test_population_catalog_profile_stream_copy_reset_and_timing_key(factory,monkeypatch):
    monkeypatch.setattr(population_timing,'materialize_visit_timing',lambda value,source:value)
    calls=[]
    def offsets(model,key,entry,prior,rng):
        calls.append((key,entry));return [86400.,8*86400.,15*86400.,22*86400.],'fixture'
    monkeypatch.setattr(main,'sample_session_offsets',offsets)
    env=factory(edit=population_edit,horizon_days=14)
    products=copy.deepcopy(env.product_rows)
    env.reset(7);profile=copy.deepcopy(env.profile);groups=copy.deepcopy(env.home_groups)
    assert calls and all(key=='buyer' for key,_ in calls)
    assert env.component_scope=='synthetic_population_group'
    assert set(profile['product_weights'])<=env.available_products
    # Synthetic baskets are sampled with replacement, rather than a source customer's finite bank.
    for _ in range(7):assert sum(env._basket().values())==2
    env.reset(7)
    assert env.profile==profile and env.home_groups==groups and env.product_rows==products
    assert env._basket_banks=={}
    left,right=copy.deepcopy(env),copy.copy(env)
    assert left.home_groups is not right.home_groups
    assert left.profile_rng is not right.profile_rng
    left.advance(0,7);right.advance(0,7)
    pd.testing.assert_frame_equal(left.tables['events'],right.tables['events'])
    assert left.rng.bit_generator.state==right.rng.bit_generator.state


def test_fixed_critic_keeps_default_policy_and_only_shifts_purchase_odds(factory):
    native=factory(horizon_days=14)
    config=copy.deepcopy(native.config);config['source_simulation']['choice_model']='fixed_irl_critic'
    env=main.MainCampaignEnv(config,warmup_days=30,min_history_days=0,horizon_days=14)
    env.reset(3)
    np.testing.assert_array_equal(env._policy,env.latent['component_policies'][env.component])
    np.testing.assert_array_equal(env.theta[:4],np.zeros(4))
    assert env.choice_space and env.solve_policy is None and env.gmm is None
    base=np.asarray([[.2,.1,0.,.3,.4],[0.,0.,0.,0.,1.]])
    env._policy=base;env._shifted_policies={0.:base}
    env.runtime_reward_offsets=lambda *_:{'purchase_choice_logit_shift':2.}
    # Use a complete action distribution just to inspect the policy compilation.
    env._active={'decisions':0,'last_at':env.now,'window_upper':env.episode_end,'next_start':None}
    env._context=lambda session:{}
    original=main.shopping.policy
    try:
        main.shopping.policy=lambda model,ctx,comp,probabilities=None:dict(zip(main.shopping.ACTIONS,[0,0,0,0,1]))
        env._close_session=lambda reason:None
        env._plan()
    finally:main.shopping.policy=original
    expected=base.copy();expected[:,3]*=np.exp(2.);expected/=expected.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(env._shifted_policies[2.],expected,atol=1e-15)
    np.testing.assert_array_equal(env._shifted_policies[0.],base)
    assert env._shifted_policies[2.][0,2]==0. and env.solve_policy is None


def test_conditioned_first_observed_entry_reanchors_without_resampling(factory,monkeypatch):
    native=factory(horizon_days=14)
    config=copy.deepcopy(native.config)
    config['source_simulation'].update(choice_model='fixed_irl_critic',source_observation_mode='first_observed_entry_conditioned')
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([86400.,3*86400.],'fixture'))
    env=main.MainCampaignEnv(config,warmup_days=0,min_history_days=0,horizon_days=14)
    env._reset_candidate(5,0)
    assert env._window['potential']==[env.origin,env.origin+pd.Timedelta(days=2)]


def test_private_conditioning_matches_joint_home_actor_given_role(factory,monkeypatch):
    monkeypatch.setattr(population_timing,'materialize_visit_timing',lambda value,source:value)
    def edit(model):
        population_edit(model)
        a=model['population']['home_groups']['buyer'];a.update(customers=3,component_probabilities=[.2,.8])
        b=copy.deepcopy(a);b.update(customers=1,component_probabilities=[.8,.2])
        model['population']['home_groups']['other']=b
    native=factory(edit=edit,warmup_days=0,horizon_days=14)
    conditioned=main.MainCampaignEnv(native.config,warmup_days=0,min_history_days=0,horizon_days=14,
        component_conditioning=[.1,.9])
    # P(G,K|R) is proportional to original P(G)P(K|G)P(R|K).
    expected=np.array([[.75*.2*.1,.75*.8*.9],[.25*.8*.1,.25*.2*.9]])
    expected/=expected.sum()
    counts=np.zeros((2,2))
    for seed in range(1000):
        conditioned._reset_candidate(seed,0)
        g=0 if conditioned.profile['home_group']=='buyer' else 1
        counts[g,conditioned.component]+=1
        p=np.array([.2,.8]) if g==0 else np.array([.8,.2])
        np.testing.assert_allclose(conditioned.profile['component_probabilities'],p*np.array([.1,.9])/(p@np.array([.1,.9])))
    np.testing.assert_allclose(counts/counts.sum(),expected,atol=.03,rtol=0)
    explicit=main.MainCampaignEnv(native.config,warmup_days=0,min_history_days=0,horizon_days=14,component_conditioning=None)
    native._reset_candidate(21,0);explicit._reset_candidate(21,0)
    assert native.profile==explicit.profile and native.component==explicit.component
    assert native.rng.bit_generator.state==explicit.rng.bit_generator.state
