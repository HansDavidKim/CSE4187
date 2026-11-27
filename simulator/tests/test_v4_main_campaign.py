"""Native Main scheduling, phase/history persistence and factual accounting."""
import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from simulate.v4.features import EXPANDED_FEATURES, build_features
from simulate.v4.irl.fit_return import FEATURE_NAMES
from simulate.v4.irl.return_state import make_model
from simulate.v4.simulator import main_campaign as main
from simulate.v4.simulator.raw_store import RAW_COLUMNS, validate_raw_dataset
from test_v4_checkout_simulator import cart_fulfillment


@pytest.fixture
def factory(tmp_path, monkeypatch):
    def write(name, value):
        path=tmp_path/name
        path.write_text(json.dumps(value))
        return str(path)
    profile={'profile_id':'p0','home_group':'buyer','preferred_price_minor':1000,
        'product_weights':{'A':1},'choice_gap_seconds':[0.],
        'checkout_templates':[{'items':[{'product_id':'A','quantity':1}],'observed_count':4}]}
    base={'currency':'source_unit','minor_unit':.01,'reference_price_minor':1000,
        'profiles':[profile],'home_groups':{'buyer':{}},
        'catalog':[{'product_id':'A','category_id':'a','unit_price_minor':1000}],
        'source_contract':{'domain':'cosmetics'}, 'latent_shopping':{
            'mixture_weights':[.5,.5],'profile_posteriors':{'p0':[0.,1.]},
            'encoder':{},'component_offsets':[[0.]*4,[0.]*4],
            'common_reward':[[0.]*4]*2,'component_values':[[0.,0.]]*2,
            'component_policies':[[[1.,0,0,0,0],[0.,0,0,0,1.]]]*2,
            'action_mask':[[True,False,False,False,False],[False,False,False,False,True]],
            'return_activity':{'component_logit_offsets':[0.,0.]}}}
    count=len(FEATURE_NAMES)
    returns={'schema_version':1,'horizon_days':30,'feature_names':FEATURE_NAMES,
        'scaler_mean':[0.]*count,'scaler_scale':[1.]*count,'feature_min':[0.]*count,
        'feature_max':[100.]*count,'coef':[0.]*count,'intercept':20.,'cold_start_probability':1.}
    monkeypatch.setattr(main,'validate_joint',lambda *a,**kw:None)
    monkeypatch.setattr(main,'validate_timing',lambda *a:None)
    monkeypatch.setattr(main.shopping,'sample_component',lambda *args:(1,'fixture'))
    monkeypatch.setattr(main.shopping,'unpack_matrices',lambda block:[sparse.csr_matrix((2,2)) for _ in range(5)])
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([86400.,8*86400.,15*86400.,22*86400.],'fixture'))
    def policy(model,context,component,probabilities=None):
        action='view' if context['session_decision_count']==0 else 'end'
        return {key:float(key==action) for key in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    def make(*, edit=None, gmm=None, eta=None, choice=False, **kwargs):
        model=copy.deepcopy(base)
        if edit:edit(model)
        model_path=write('shopping.json',model)
        return_path=write('return.json',returns)
        settings={'model_path':model_path,'return_model_path':return_path,
                  'visit_timing_model_path':write('timing.json',{'source_domain':model['source_contract']['domain']}),'max_events_per_session':8}
        if eta is not None:settings['return_state_model_path']=write('eta.json',eta)
        if gmm is not None:
            artifact={'schema_version':1,'type':'source_persona_joint_reward_gmm','purpose':'synthesis_only',
                'parameter_names':['view','cart','remove','purchase','return_logit'],
                'component_count':2,'mixture_weights':[.5,.5], 'means':[[.1,.2,.3,.4,0.],[.5,.6,.7,.8,.9]],
                'covariances':np.repeat(np.eye(5)[None],2,axis=0).tolist(),
                'component_model_indices':[0,1],'persona_labels':['P0','P1'],
                'condition_on_profile_components':True,'dispersion_scales':[0.,0.],
                'base_model_sha256':hashlib.sha256((tmp_path/'shopping.json').read_bytes()).hexdigest(),
                'return_model_sha256':hashlib.sha256((tmp_path/'return.json').read_bytes()).hexdigest()}
            if choice:
                from simulate.v4.irl import choice_gmm
                artifact.update(type=choice_gmm.TYPE,backend=choice_gmm.BACKEND,
                    parameter_names=choice_gmm.PARAMETERS,parameter_units=choice_gmm.UNITS,
                    fitted_parameter_names=choice_gmm.PARAMETERS)
            artifact.update(gmm)
            settings['choice_gmm_model_path' if choice else 'reward_gmm_model_path']=write('gmm.json',artifact)
        config={'simulation':{'start_at':'2024-01-01T00:00:00+09:00'},'source_simulation':settings}
        return main.MainCampaignEnv(config,warmup_days=kwargs.pop("warmup_days",30),min_history_days=0,**kwargs)
    return make


def oracle(env):
    values=build_features(env.tables,as_of_at=env.now).iloc[0][list(EXPANDED_FEATURES)].to_numpy(float)
    np.testing.assert_allclose(env._features(),values,equal_nan=True)


@pytest.mark.parametrize("held,quantity,source,remaining", [
    ({"A": 2, "B": 1}, 1, "cart", {"A": 1, "B": 1}),
    ({"A": 1, "B": 1}, 2, "mixed", {"B": 1}),
    ({"B": 1}, 1, "buy_now", {"B": 1}),
])
def test_checkout_uses_held_sku_and_preserves_unbought_cart(factory, monkeypatch, held, quantity, source, remaining):
    from simulate.v4.simulator.raw_store import cart_at
    from uplift.common.features import build_features as uplift_features
    def edit(model):
        model["catalog"] += [{"product_id": "template", "category_id": "a", "unit_price_minor": 1009},
                             {"product_id": "B", "category_id": "b", "unit_price_minor": 5000}]
        model["profiles"][0]["checkout_templates"] = [{
            "items": [{"product_id": "template", "quantity": quantity}], "observed_count": 1}]
    env = factory(horizon_days=7, edit=edit)
    env.reset(17)
    monkeypatch.setattr(env, "_plan", lambda: None)
    at = env.now
    env._start_session(at, env.episode_end)
    session = env._active
    tick = 0
    for pid, count in held.items():
        monkeypatch.setattr(env, "_item", lambda *args, selected=pid: selected)
        for _ in range(count):
            session["planned"] = (at + pd.Timedelta(seconds=tick), "cart", 1.)
            env._emit_action()
            tick += 1
    bought_at = at + pd.Timedelta(seconds=tick)
    session["planned"] = (bought_at, "purchase", 1.)
    env._emit_action()
    env._close_session("causal_policy_end")
    env.now = bought_at + pd.Timedelta(minutes=31)
    tables = env.tables
    assert tables["orders"].iloc[-1].checkout_source == source
    purchased = tables["order_items"].groupby("product_id").quantity.sum().to_dict()
    assert sum(purchased.values()) == quantity
    assert purchased.get("A", 0) == min(held.get("A", 0), quantity)
    assert dict(+env.state["cart"]) == cart_at(tables, env.uid, env.now) == remaining
    assert not tables["events"].event_type.eq("remove_from_cart").any()
    validate_raw_dataset({name: tables[name] for name in RAW_COLUMNS}, env.records["_metadata"])
    kit = build_features(tables, as_of_at=env.now).iloc[0]
    peer = uplift_features(tables, pd.DataFrame({"customer_id": [env.uid], "t0": [env.now]}), env.origin).iloc[0]
    for feature in ("cart_qty_current", "cart_sku_current", "cart_gross_value_current"):
        assert kit[feature] == peer[feature]
    oracle(env)
    # An actual later remove action still changes the held inventory by one unit.
    env._start_session(env.now, env.episode_end)
    monkeypatch.setattr(env, "_item", lambda *args: "B")
    env._active["planned"] = (env.now, "remove", 0.)
    env._emit_action()
    env.now += pd.Timedelta(nanoseconds=1)
    remaining = {pid: count for pid, count in remaining.items() if pid != "B"}
    assert dict(+env.state["cart"]) == cart_at(env.tables, env.uid, env.now) == remaining
    assert env.records["events"][-1]["event_type"] == "remove_from_cart"


def test_checkout_fitted_cart_quota_balances_actual_prices_without_clearing_stock(factory, monkeypatch, cart_fulfillment):
    from simulate.v4.irl.fit_checkout_spending import TYPE
    from simulate.v4.simulator import population_profiles
    from simulate.v4.simulator.raw_store import cart_at
    def edit(model):
        model["catalog"] += [dict(product_id=pid, category_id="fixture", unit_price_minor=price)
                             for pid, price in (("target", 600), ("cheap", 500), ("dear", 700))]
        model["profiles"][0]["checkout_templates"] = [{
            "items": [{"product_id": "target", "quantity": 4}], "observed_count": 1}]
    env = factory(horizon_days=7, edit=edit)
    env.reset(17)
    preference = np.log([4., 600.]).tolist()
    env.population = {"checkout_spending": {"schema_version": 1, "type": TYPE, "mean": preference,
        "between_covariance": [[0., 0.], [0., 0.]], "within_covariance": [[1., 0.], [0., 1.]],
        "cart_fulfillment": cart_fulfillment}}
    env.profile["checkout_spending_preference"] = preference
    monkeypatch.setattr(env, "_plan", lambda: None)
    at = env.now
    env._start_session(at, env.episode_end)
    session = env._active
    for tick, pid in enumerate(["cheap"] * 4 + ["dear"] * 2):
        monkeypatch.setattr(env, "_item", lambda *args, selected=pid: selected)
        session["planned"] = (at + pd.Timedelta(seconds=tick), "cart", 1.)
        env._emit_action()
    allocate, calls = population_profiles._budgeted_cart_units, []
    def budgeted(*args):
        calls.append(True)
        return allocate(*args)
    monkeypatch.setattr(population_profiles, "_budgeted_cart_units", budgeted)
    session["planned"] = (at + pd.Timedelta(seconds=6), "purchase", 1.)
    env._emit_action()
    env._close_session("causal_policy_end")
    env.now = at + pd.Timedelta(minutes=31)
    tables = env.tables
    assert calls == [True]
    assert tables["orders"].iloc[-1].checkout_source == "cart"
    assert tables["orders"].iloc[-1].gross_minor == tables["orders"].iloc[-1].paid_minor == 2400
    assert tables["order_items"].groupby("product_id").quantity.sum().to_dict() == {"cheap": 2, "dear": 2}
    assert dict(+env.state["cart"]) == cart_at(tables, env.uid, env.now) == {"cheap": 2}
    assert not tables["events"].event_type.eq("remove_from_cart").any()
    validate_raw_dataset({name: tables[name] for name in RAW_COLUMNS}, env.records["_metadata"])
    oracle(env)


def test_native_banks_and_raw_history_identical_for_whole_weekly_daily_advances(factory,monkeypatch):
    calls=[]
    def offsets(*args):
        calls.append(args[2:4]);return [86400.,8*86400.,15*86400.,22*86400.],'fixture'
    monkeypatch.setattr(main,'sample_session_offsets',offsets)
    whole,sliced=factory(horizon_days=35),factory(horizon_days=35)
    whole.reset(7);sliced.reset(7)
    assert len(calls)==2
    whole.advance(0,35)
    for _ in range(35):sliced.advance(0,1);oracle(sliced)
    assert len(calls)==6
    for name in (*RAW_COLUMNS,'session_diagnostics','lifecycle'):
        pd.testing.assert_frame_equal(whole.tables[name],sliced.tables[name])
    assert whole.state['observed_session_number']==len(whole.records['sessions'])
    assert whole.state['cart']==sliced.state['cart']
    validate_raw_dataset({k:v for k,v in whole.tables.items() if k in RAW_COLUMNS},whole.records['_metadata'])


def test_natural_end_precedes_gap_draw_and_tied_starts_are_not_removed(factory,monkeypatch):
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([86400.,86400.,86400.,22*86400.],'fixture'))
    env=factory(horizon_days=3);env.reset(8)
    assert len(env.records['sessions'])==4
    assert env.records['sessions'][0]['started_at']==env.records['sessions'][2]['started_at']
    assert all(row['termination_reason']=='causal_policy_end' for row in env.records['session_diagnostics'])
    monkeypatch.setattr(env,'_sample',lambda *args:pytest.fail('END must not draw latency'))
    env.advance(0,3)
    oracle(env)


def test_private_phase_count_persists_after_exit_and_t0(factory,monkeypatch):
    contexts=[]
    def policy(model,c,z,probabilities=None):
        contexts.append(dict(c));action='cart' if c['session_decision_count']==0 else 'end'
        return {key:float(key==action) for key in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([3*86400.],'fixture'))
    env=factory(horizon_days=65);env.reset(9)
    before=env.state['observed_session_number'];cart=env.state['cart']['A'];theta=env.theta.copy()
    env.advance(0,65)
    assert before==1 and env.state['observed_session_number']==4
    assert [c['observed_session_number'] for c in contexts if c['session_decision_count']==0]==[1,2,3,4]
    assert env.state['cart']['A']==cart+3
    np.testing.assert_array_equal(env.theta,theta)
    assert set(env.tables['lifecycle'].life_cycle)=={'ENTRY','HOME','EXIT'}
    assert env.tables['lifecycle'].as_of_at.lt(env.now).all()
    oracle(env)


def test_pending_choice_cannot_be_changed_by_future_coupon_and_copy_isolated(factory,monkeypatch):
    decisions=[]
    def policy(model,c,z,probabilities=None):
        decisions.append((c['session_decision_count'],float(probabilities[0,3])))
        action=('cart','purchase','end')[min(2,c['session_decision_count'])]
        return {key:float(key==action) for key in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    # One potential visit per30d: the pending purchase spans both warmup t0 and a weekly boundary.
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([8*86400.],'fixture'))
    def shifts(env,at):
        return {'purchase_reward_shift':2. if env._receipts else 0.}
    def response(env,at):return {'response_selected':True,'response_ready_at':at}
    env=factory(horizon_days=14,warmup_days=14,edit=lambda m:m['profiles'][0].update(choice_gap_seconds=[20*86400.]),
        response_sampler=response,runtime_reward_offsets=shifts)
    env.reset(10);assert env._active['planned'][1]=='purchase'
    clone=copy.copy(env)
    assert clone._active['row'] is clone.records['sessions'][-1]
    assert clone._active['row'] is not env._active['row']
    assert clone._active['diagnostic'] is clone.records['session_diagnostics'][-1]
    chosen=len(decisions)
    clone.advance(1,7)
    assert len(decisions)==chosen  # not emitted yet; no retroactive resampling
    assert not env._receipts and not env.records['coupons']
    state,reward,done,info=clone.advance(1,7)
    assert done and reward==900 and info['orders']==info['converted_sessions']==info['redemptions']==1
    assert info['discount_minor']==100 and info['paid_minor']==900
    assert clone.records['orders'][0]['coupon_id']==f'{clone.uid}:C1'
    assert clone._receipts[0]['used'] and not clone._receipts[1]['used']
    assert env._active['row']['ended_at'] is None
    oracle(clone)
    validate_raw_dataset({k:v for k,v in clone.tables.items() if k in RAW_COLUMNS},clone.records['_metadata'])


def test_max_events_is_fatal_and_not_an_initial_candidate_retry(factory,monkeypatch):
    monkeypatch.setattr(main.shopping,'policy',lambda m,c,z,probabilities=None:{a:float(a=='view') for a in main.shopping.ACTIONS})
    env=factory(horizon_days=7);env.max_events=2
    with pytest.raises(RuntimeError,match='cap reached'):env.reset(11)
    assert env.records['session_diagnostics'][-1]['termination_reason']=='max_events'
    assert env.records['session_diagnostics'][-1]['truncated_session']
    assert len(env.records['sessions'])==1
    with pytest.raises(RuntimeError,match='session cap'):env.advance(0)


def test_source_profile_gaussian_draw_persists_and_is_not_discrete_offset_double_counted(factory):
    env=factory(gmm={},horizon_days=35)
    env.reset(12)
    assert env.gmm_component==env.component==1
    np.testing.assert_array_equal(env.theta,[.5,.6,.7,.8,.9])
    expected=main._logit(main.return_probability(env.return_model,
        main.return_context([],[],env.origin,purchase_item_units={})))+.9
    assert env._window['logit']==pytest.approx(expected)
    theta=env.theta.copy();env.advance(0,35)
    np.testing.assert_array_equal(env.theta,theta)
    oracle(env)


def test_eta_zero_reproduces_base_stream_and_only_natural_completions_advance(factory):
    base=factory(horizon_days=35);zero=factory(horizon_days=35,eta=make_model(.8,0.))
    base.reset(13);zero.reset(13)
    base.advance(0,35);zero.advance(0,35)
    for name in RAW_COLUMNS:pd.testing.assert_frame_equal(base.tables[name],zero.tables[name])
    assert zero.state['return_state'].eta==0
    assert zero.state['return_state'].updates==len(zero.records['sessions'])
    assert zero.state['return_state'].rng.bit_generator.state==np.random.default_rng(int.from_bytes(
        hashlib.sha256(json.dumps(['source-event-stream-v1',13,f'{zero.uid}:0','return_state']).encode()).digest()[:8],'big')).bit_generator.state


@pytest.mark.parametrize("choice", [False, True])
def test_return_state_requires_its_frozen_static_gmm(factory, choice):
    original = factory(gmm={}, choice=choice)
    fingerprint = hashlib.sha256(json.dumps(original.gmm,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    eta = make_model(.8, .2)
    eta["fit"] = {"static_return_prior": {"population_model_fingerprint": fingerprint}}
    with pytest.raises(ValueError, match="requires its persistent reward GMM"):
        factory(eta=eta)
    mismatch = copy.deepcopy(eta)
    mismatch["runtime_gmm_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="exact frozen static reward GMM"):
        factory(gmm={}, choice=choice, eta=mismatch)
    bound = factory(gmm={}, choice=choice, eta=eta)
    bound.reset(13)
    assert bound.state["return_state"].model == eta
    # Aggregate exports bind the runtime GMM while retaining the source-fit fingerprint.
    eta["fit"]["static_return_prior"]["population_model_fingerprint"] = "1" * 64
    eta["runtime_gmm_fingerprint"] = fingerprint
    factory(gmm={}, choice=choice, eta=eta).reset(13)


def test_no_coupon_defaults_and_domain_mismatch_fail_explicitly(factory):
    env=factory(horizon_days=7);env.reset(14)
    with pytest.raises(ValueError,match='response_sampler'):env.advance(1)
    assert not env.records['coupons']
    electronics=factory(horizon_days=7,edit=lambda m:m.update(source_contract={'domain':'electronics'},
        catalog=[{'product_id':'E','category_id':'e','unit_price_minor':7351}],
        profiles=[{**m['profiles'][0],'product_weights':{'E':1},'checkout_templates':[
            {'items':[{'product_id':'E','quantity':1}],'observed_count':4}]}]))
    electronics.reset(14)
    assert electronics.domain=='electronics' and electronics.product_prices=={'E':7351}
    assert {e['product_id'] for e in electronics.records['events']}=={'E'}
    with pytest.raises(ValueError,match='matching'):
        factory(edit=lambda m:m['latent_shopping'].update(source_domain='electronics'))


def test_native_window_and_scheduled_next_start_censor_keep_observed_prefix(factory,monkeypatch):
    def policy(model,c,z,probabilities=None):
        action=('cart','purchase','end')[min(2,c['session_decision_count'])]
        return {a:float(a==action) for a in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([8*86400.,9*86400.],'fixture'))
    env=factory(horizon_days=4,edit=lambda m:m['profiles'][0].update(choice_gap_seconds=[25*86400.]))
    env.reset(16)
    assert [d['termination_reason'] for d in env.records['session_diagnostics']]==['scheduled_next_start','window_end']
    assert len(env.records['events'])==2 and env.state['cart']['A']==2
    assert all(d['truncated_session'] for d in env.records['session_diagnostics'])
    oracle(env)
    short=factory(horizon_days=4,edit=lambda m:m['profiles'][0].update(choice_gap_seconds=[2*86400.]))
    short.reset(16)
    assert short.records['session_diagnostics'][0]['termination_reason']=='scheduled_next_start'
    assert short.records['events'][0]['event_type']=='cart'
    oracle(short)


def test_coupon_purchase_shift_resolves_bellman_at_decision_time(factory,monkeypatch):
    calls=[]
    def prepared(latent,matrices):
        def solve(offsets,**kwargs):
            calls.append((np.array(offsets),kwargs))
            return np.array([[1.,0,0,0,0],[0,0,0,0,1.]])
        return solve
    monkeypatch.setattr(main.reward_gmm,'prepare_bellman',prepared)
    def response(env,at):return {'response_selected':False,'response_ready_at':None}
    def shifts(env,at):return {'purchase_reward_shift':1.25 if env._receipts else 0.}
    env=factory(horizon_days=14,response_sampler=response,runtime_reward_offsets=shifts)
    env.reset(17);env.advance(1,14)
    assert len(calls)==1
    np.testing.assert_allclose(calls[0][0],env.theta[:4]+[0,0,0,1.25])
    assert calls[0][1]==dict(component_id=1,temperature=1.)


def test_private_return_updates_wait_strictly_for_known_end(factory):
    env=factory(horizon_days=7,eta=make_model(.7,.5))
    env.reset(18)
    private=env.state['return_state']
    assert private.updates==len(env.records['sessions'])
    # Snapshot reads and an EXIT timer do not themselves evolve the state.
    before=private.rng.bit_generator.state
    eta=private.eta
    env._features();env.tables
    assert private.eta==eta and private.rng.bit_generator.state==before


def test_S_D_costs_only_change_score_and_do_not_leak_future_into_eligibility(factory):
    response=lambda env,at:{'response_selected':False,'response_ready_at':None}
    s=factory(cost_model='S',horizon_days=7,response_sampler=response)
    d=factory(cost_model='D',horizon_days=7,response_sampler=response)
    s.reset(19);d.reset(19)
    short=factory(horizon_days=1);short.reset(19)
    assert s.source_profile_id==short.source_profile_id and s.records['sessions']==short.records['sessions']
    assert s.advance(1)[1]==-100 and d.advance(1)[1]==0
    for name in RAW_COLUMNS:pd.testing.assert_frame_equal(s.tables[name],d.tables[name])


@pytest.mark.parametrize('cost_model',['S','D'])
def test_objective_cost_changes_reward_without_changing_retail_paths(factory,monkeypatch,cost_model):
    def policy(model,context,component,probabilities=None):
        action=('cart','purchase','end')[min(context['session_decision_count'],2)]
        return {key:float(key==action) for key in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    response=lambda env,at:{'response_selected':True,'response_ready_at':at}
    old=factory(cost_model=cost_model,horizon_days=14,response_sampler=response)
    high=factory(cost_model=cost_model,horizon_days=14,response_sampler=response,score_coupon_cost=500.)
    np.testing.assert_allclose(old.reset(191),high.reset(191),equal_nan=True)
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
        purchases+=info['converted_sessions'];uses+=info['redemptions']
    assert purchases>0 and uses>0
    for name in (*RAW_COLUMNS,'runtime_lifecycle','session_diagnostics'):
        pd.testing.assert_frame_equal(old.tables[name],high.tables[name])
    assert old.rng.bit_generator.state==high.rng.bit_generator.state
    assert high.records['_metadata']['score_coupon_cost']==500


@pytest.mark.parametrize('name,value',[
    ('score_conversion_value',0),('score_conversion_value',True),('score_conversion_value',np.inf),
    ('score_conversion_value','1000'),('score_coupon_cost',-1),('score_coupon_cost',np.bool_(True)),
    ('score_coupon_cost',np.nan),('score_coupon_cost','500'),
])
def test_retail_objective_rejects_invalid_weights(factory,name,value):
    with pytest.raises(ValueError,match=name):factory(**{name:value})


def test_private_churning_exit_and_coupon_reentry_keep_same_history_and_theta(factory,monkeypatch):
    monkeypatch.setattr(main,'sample_session_offsets',lambda *args:([3*86400.],'fixture'))
    def policy(model,c,z,probabilities=None):
        action='cart' if c['session_decision_count']==0 else 'end'
        return {a:float(a==action) for a in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    def shifts(env,at):
        return {'return_logit_shift':80. if any(r['sent_at']<=at<r['expires_at'] for r in env._receipts) else 0.}
    response=lambda env,at:{'response_selected':False,'response_ready_at':None}
    env=factory(horizon_days=66,response_sampler=response,runtime_reward_offsets=shifts)
    env.return_model['intercept']=-40.
    env.reset(20)
    theta=env.theta.copy();initial_cart=env.state['cart']['A']
    env.advance(0,1)
    bank=env._window;draw=bank['draw'];potential=list(bank['potential'])
    assert env._lifecycle_phase=='EXIT'
    assert 'CHURNING' in [r['phase'] for r in env.records['runtime_lifecycle']]
    env.advance(1,7)
    assert env._window is bank and bank['draw']==draw and bank['potential']==potential
    assert env._lifecycle_phase=='NEW' and env.state['observed_session_number']==2
    assert env.state['cart']['A']==initial_cart+1
    env.advance(0,30)
    assert env._lifecycle_phase=='EXIT'
    env.advance(1,28)
    assert env._lifecycle_phase=='HOME' and env.state['observed_session_number']==3
    assert env.state['cart']['A']==initial_cart+2
    np.testing.assert_array_equal(env.theta,theta)
    rows=env.records['runtime_lifecycle']
    assert any(r['reason']=='actual_START_return_after_EXIT' and r['phase']=='HOME' for r in rows)
    assert set(env.tables['lifecycle'].life_cycle)<= {'ENTRY','HOME','EXIT'}
    assert 'CHURNING' not in EXPANDED_FEATURES and env._features().shape==(39,)
    oracle(env)


def test_internal40_matches_independent_source_replay_without_filling_unseen_view_price(factory,monkeypatch):
    contexts=[]
    def policy(model,c,z,probabilities=None):
        contexts.append(dict(c))
        action=('cart','view','purchase','end')[min(3,c['session_decision_count'])]
        return {a:float(a==action) for a in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    def edit(model):
        model['catalog'].append({'product_id':'B','category_id':'b','unit_price_minor':7000})
        model['profiles'][0]['checkout_templates']=[{'items':[{'product_id':'B','quantity':2}],'observed_count':4}]
    env=factory(horizon_days=14,edit=edit)
    env.reset(21);env.advance(0,14)
    # Only a view supplies the MDP's offer-price coordinate. Cart/purchase prices
    # are raw observations but cannot replace the last-view state variable.
    assert [c['unit_price_minor'] for c in contexts[:4]]==[None,None,1000,1000]
    by_order={o['purchase_event_id']:o['order_id'] for o in env.records['orders']}
    baskets={oid:[i for i in env.records['order_items'] if i['order_id']==oid] for oid in by_order.values()}
    rows=[]
    for event in env.records['events']:
        items=(baskets[by_order[event['event_id']]] if event['event_type']=='purchase'
            else [dict(product_id=event['product_id'],quantity=1,unit_price_minor=event['unit_price_minor'])])
        for item in items:
            for _ in range(item['quantity']):
                rows.append(dict(user_id=env.uid,session=int(event['session_id'].rsplit('S',1)[1]),
                    action=event['event_type'],product_id=item['product_id'],price_minor=item['unit_price_minor'],
                    at=event['event_at'],source_event_id=f'{len(rows):09d}'))
    replay=main.shopping.build_source(pd.DataFrame(rows),env.now)
    actual=np.asarray([main.shopping.context_vector(c) for c in contexts])
    np.testing.assert_allclose(actual,replay['features'],rtol=1e-12,atol=1e-12)
    oracle(env)


def test_reward_hook_observes_actual_decision_clock_and_strict_raw39(factory):
    seen=[]
    def shifts(env,at):
        assert env.now==at
        expected=build_features(env.tables,as_of_at=at).iloc[0][list(EXPANDED_FEATURES)].to_numpy(float)
        np.testing.assert_allclose(env._features(),expected,equal_nan=True)
        seen.append(at)
        return {'purchase_reward_shift':0.,'return_logit_shift':0.}
    env=factory(horizon_days=35,runtime_reward_offsets=shifts)
    env.reset(22);env.advance(0,35)
    assert seen==sorted(seen) and len(set(seen))>8


def test_purchase_natural_end_records_mission_exit_then_same_persona_reentry(factory,monkeypatch):
    seen=[]
    def policy(model,c,z,probabilities=None):
        seen.append((z,c['observed_session_number']))
        action='purchase' if c['session_decision_count']==0 else 'end'
        return {a:float(a==action) for a in main.shopping.ACTIONS}
    monkeypatch.setattr(main.shopping,'policy',policy)
    env=factory(horizon_days=14,response_sampler=lambda env,at:{'response_selected':True,'response_ready_at':at})
    env.reset(23);theta=env.theta.copy();actor=env.component
    env.advance(1,14)
    assert sum(row['phase']=='PURCHASE_EXIT' for row in env.records['runtime_lifecycle'])==len(env.records['sessions'])
    assert any(row['reason']=='actual_START_return_after_PURCHASE_EXIT' for row in env.records['runtime_lifecycle'])
    assert all(z==actor for z,number in seen)
    assert max(number for z,number in seen)>2
    np.testing.assert_array_equal(theta,env.theta)
    assert all(d['termination_reason']=='causal_policy_end' and not d['truncated_session'] for d in env.records['session_diagnostics'])
    assert set(env.tables['lifecycle'].life_cycle)<={'ENTRY','HOME','EXIT'}


def test_separate_choice_gmm_keeps_irl_representative_without_customer_bellman_replan(factory,monkeypatch):
    from simulate.v4.irl import choice_gmm
    monkeypatch.setattr(main.reward_gmm,'prepare_bellman',lambda *a:pytest.fail('choice tastes must not replan a customer MDP'))
    shifts=lambda env,at:{'purchase_choice_logit_shift':.3,'return_logit_shift':0.}
    env=factory(choice=True,gmm={},horizon_days=14,runtime_reward_offsets=shifts)
    env.reset(24)
    assert env.choice_space and env.component==1 and env.purchase_shift_name=='purchase_choice_logit_shift'
    np.testing.assert_allclose(env.theta,[.5,.6,.7,.8,.9])
    expected=choice_gmm.prepare_choice_policy(env.latent)(env.theta[:4]+[0,0,0,.3],component_id=1,temperature=1.)
    np.testing.assert_array_equal(env._shifted_policies[.3],expected)
    theta=env.theta.copy();env.advance(0,14)
    np.testing.assert_array_equal(env.theta,theta)
    assert env.records['_metadata']['shopping_offset_units']=='choice_logit_shift'
    oracle(env)


def test_choice_gmm_rejects_reward_units_and_mixed_sidecar_paths(factory):
    env=factory(choice=True,gmm={},runtime_reward_offsets=lambda env,at:{'purchase_reward_shift':.3})
    with pytest.raises(ValueError,match='purchase_choice_logit_shift'):env.reset(25)
    config=copy.deepcopy(env.config)
    config['source_simulation']['reward_gmm_model_path']=config['source_simulation']['choice_gmm_model_path']
    with pytest.raises(ValueError,match='one continuous'):main.MainCampaignEnv(config)
