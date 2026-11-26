"""Toy-only runner proof; no fitted source world or production policy is trained."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from models.agents import AGENTS
from scripts import main_world_benchmark_utils as utils
from scripts import run_main_world_benchmark as runner


class ToyWeeklyEnv:
    reset_calls=0
    def __init__(self,weeks=3,domain='cosmetics',cost='S',conversion_value=1000.,coupon_cost=100.):
        self.weeks=weeks;self.domain=domain;self.cost_model=cost
        self.score_conversion_value=conversion_value;self.score_coupon_cost=coupon_cost
        self.horizon_days=weeks*7;self.warmup_days=364;self.min_history_days=91;self.coupon_valid_days=28
        self.config={'simulation':{'toy':True},'domain':domain,'weeks':weeks};self.settings={}
        self.home_groups={'constructor':'not an episode'}
        self.model_shas={'toy':'fixed'};self.profiles=({'profile_id':'p0'},);self.profile_probabilities=np.array([1.])
        self.product_rows=({'product_id':'A','unit_price_minor':1000},)
        self.feature_builder=self.response_sampler=self.runtime_reward_offsets=None
    def reset(self,seed):
        type(self).reset_calls+=1
        self.home_groups={'seed':seed,'templates':[seed%11]}
        self.rng=np.random.default_rng(seed);self.step=0;self.previous=0;self.base=seed%17/17
        first={'seed':seed}
        self.records={'sessions':[first],'products':list(self.product_rows),'_metadata':{'initial_candidate_attempts':1,'cost_model':self.cost_model}}
        self.records['_metadata'].update(score_conversion_value=self.score_conversion_value,score_coupon_cost=self.score_coupon_cost)
        self._active={'row':first};self.state={'rng':self.rng}
        return self._features()
    def _features(self):
        state=np.zeros(39);state[:3]=self.base,self.step,self.previous;state[-1]=np.nan
        return state
    def advance(self,action,days=7):
        assert days==7
        shock=self.rng.random();converted=int(shock<.2+.45*action+.1*self.previous)
        self.previous=action;self.step+=1
        self.records['sessions'].append({'shock':shock,'action':action})
        info={'converted_sessions':converted,'orders':converted,'sends':action,'redemptions':action*converted,
            'paid_minor':900*converted,'discount_minor':100*action*converted,'send_cost_minor':100*action}
        if self.domain=='tianchi':
            for name in utils.UNOBSERVED_T:info[name]=None
        cost=info['sends'] if self.cost_model=='S' else info['redemptions']
        return self._features(),self.score_conversion_value*converted-self.score_coupon_cost*cost,self.step==self.weeks,info


class PatternPolicy:
    def __init__(self,pattern):self.pattern=pattern
    def act(self,state,training=False):
        assert not training
        return self.pattern[int(state[1])%len(self.pattern)]
    def update(self,*args):raise AssertionError('frozen evaluation cannot update a policy')


def normalizer():return {'median':[0.]*39,'mean':[0.]*39,'scale':[1.]*39,'clip':100.}


def job(domain='cosmetics'):
    return {'key':'toy','n_train':2,'train_weeks':3,'n_test':7,'test_weeks':3,'eval_batch_size':3,
        'domain':domain,'seed':42,'evaluation_seed':95001,'learning_starts':4,'sac_batch_size':4,
        'ppo_rollout':4,'progress_users':1,'initial_state_cache':None}


def test_plan_has42_cells_and_original_budget_without_constructing_worlds(tmp_path):
    options={'domains':list(runner.DOMAINS),'costs':['S','D'],'agents':list(AGENTS),'seed':42,
        'n_train':1000,'train_weeks':13,'n_test':5000,'test_weeks':52,'calibration_users':128,'workers':2,'device':'mps'}
    jobs=runner.build_jobs(options,{d:f'{d}.yaml' for d in runner.DOMAINS},tmp_path)
    assert len(jobs)==42
    assert all(j['n_train']*j['train_weeks']==13000 and j['n_test']*j['test_weeks']==260000 for j in jobs.values())
    assert all(j['device']=='mps' and j['freeze'] is None for j in jobs.values())
    assert all(j['conversion_value']==1000. and j['coupon_cost']==100. for j in jobs.values())
    expensive=runner.build_jobs({**options,'coupon_cost':500.},{d:f'{d}.yaml' for d in runner.DOMAINS},tmp_path/'high')
    assert set(expensive)==set(jobs) and all(j['coupon_cost']==500. for j in expensive.values())
    assert [j['domain'] for j in list(jobs.values())[:3]]==list(runner.DOMAINS)
    assert runner.evaluation_ranges(5000,64)==[(0,2496),(2496,5000)]


def test_gate_rejects_missing_or_incomplete_freeze_before_world_construction(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'required_files',lambda path:pytest.fail('gate must fail before reading source configs'))
    with pytest.raises(ValueError,match='frozen-manifest'):runner.validate_freeze(None,{})
    path=tmp_path/'freeze.json'
    path.write_text(json.dumps({'schema_version':1,'status':'frozen','frozen_at':'2020-01-01T00:00:00+00:00','file_sha256':{'x':'fake'},'configs':{d:'unused' for d in runner.DOMAINS},'gates':{}}))
    with pytest.raises(ValueError,match='source gate'):runner.validate_freeze(path,{})


@pytest.mark.parametrize('field,bad',[(field,value) for field in ('conversion_value','coupon_cost')
    for value in (True,np.bool_(False),float('nan'),float('inf'),-1.,'500')]+[('conversion_value',0.)])
def test_score_parameters_reject_invalid_money_before_world_construction(field,bad):
    with pytest.raises(ValueError,match=field):utils.score_parameters({field:bad})


def test_evaluation_restart_requires_the_training_score_weights_and_budgets():
    saved={**job(),'calibration_users':128,'calibration_seed':7131,'coupon_cost':500.,'conversion_value':1000.}
    runner.validate_evaluation_options(saved,saved.copy())
    for field,value in [('coupon_cost',100.),('conversion_value',2000.),('n_test',8)]:
        with pytest.raises(ValueError,match='must match'):
            runner.validate_evaluation_options({**saved,field:value},saved)
    legacy={key:value for key,value in saved.items() if key not in ('coupon_cost','conversion_value')}
    runner.validate_evaluation_options({**legacy,'coupon_cost':100.,'conversion_value':1000.},legacy)


def test_training_rejects_a_different_environment_cost_before_updates(tmp_path):
    with pytest.raises(ValueError,match='score weights differ'):
        utils.train_policy(AGENTS['UCB'](39),'UCB',ToyWeeklyEnv(),normalizer(),{**job(),'coupon_cost':500.},tmp_path)


def test_gate_requires_all3_domain_evidence_and_exact_runtime_sha(tmp_path,monkeypatch):
    config=tmp_path/'config.json';config.write_text('{}')
    report=tmp_path/'report.json';report.write_text('{"status":"complete"}')
    manifest={'schema_version':1,'status':'frozen','frozen_at':'2020-01-01T00:00:00+00:00',
        'file_sha256':{str(p):utils.sha256(p) for p in (config,report)},'configs':{d:str(config) for d in runner.DOMAINS},
        'gates':{name:{'status':'complete','passed':True,'domains':list(runner.DOMAINS),'reports':[str(report)]} for name in runner.GATES}}
    freeze=tmp_path/'freeze.json';freeze.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner,'required_files',lambda path:{config})
    assert runner.validate_freeze(freeze,{'cosmetics':str(config)})['manifest_sha256']==utils.sha256(freeze)
    manifest['gates']['coupon']['domains']=['cosmetics'];freeze.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='coupon gate'):runner.validate_freeze(freeze,{})
    manifest['gates']['coupon']['domains']=list(runner.DOMAINS);freeze.write_text(json.dumps(manifest))
    config.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='input changed'):runner.validate_freeze(freeze,{'cosmetics':str(config)})


def test_private_reset_cache_preserves_rng_row_alias_and_exact_replay_across_S_D(tmp_path):
    first=ToyWeeklyEnv();expected=first.reset(7)
    cached=ToyWeeklyEnv();utils.cache_reset(cached,7,tmp_path)
    reused=ToyWeeklyEnv(cost='D',coupon_cost=500.);actual=utils.cache_reset(reused,7,tmp_path)
    np.testing.assert_array_equal(actual,expected)
    assert reused._active['row'] is reused.records['sessions'][0]
    assert reused.home_groups==cached.home_groups and reused.home_groups is not cached.home_groups
    assert reused.state['rng'] is reused.rng and reused.rng is not cached.rng
    assert reused.records['_metadata']['cost_model']=='D'
    assert reused._benchmark_last_cache_hit and reused.records['_metadata']['score_coupon_cost']==500.
    assert cached.records['_metadata']['score_coupon_cost']==100.
    for action in (1,0,1):
        first.advance(action);cached.advance(action);reused.advance(action)
    assert first.records['sessions']==cached.records['sessions']==reused.records['sessions']
    branch=utils.clone_episode(reused);branch.state['extra']=[1]
    assert 'extra' not in reused.state
    assert branch._active['row'] is branch.records['sessions'][0]
    assert branch.home_groups is reused.home_groups
    assert branch.records['products'] is not reused.records['products']
    assert branch.records['products'][0] is reused.records['products'][0]
    other=ToyWeeklyEnv();utils.cache_reset(other,7,tmp_path)
    assert other.step==0
    other.horizon_days+=7
    with pytest.raises(ValueError,match='constructor inputs changed'):utils.cache_reset(other,8,tmp_path)


@pytest.mark.parametrize('domain',('cosmetics','tianchi'))
@pytest.mark.parametrize('scoring',({},dict(conversion_value=1200.5,coupon_cost=500.25)))
def test_shared_action_prefix_matches_naive_frozen_evaluation_for_every_user(tmp_path,domain,scoring):
    j={**job(domain),**scoring}
    policies={'zeroS':{'policy':PatternPolicy([0]),'normalizer':normalizer(),'cost':'S'},
        'zeroD':{'policy':PatternPolicy([0]),'normalizer':normalizer(),'cost':'D'},
        'late':{'policy':PatternPolicy([0,1,1]),'normalizer':normalizer(),'cost':'S'},
        'early':{'policy':PatternPolicy([1,1,0]),'normalizer':normalizer(),'cost':'D'}}
    frames,protocol=utils.evaluate_shared_policies(policies,ToyWeeklyEnv(domain=domain,**scoring),j)
    for key,member in policies.items():
        directory=tmp_path/key;directory.mkdir()
        expected,_=utils.evaluate_policy(member['policy'],ToyWeeklyEnv(domain=domain,cost=member['cost'],**scoring),member['normalizer'],j,directory)
        pd.testing.assert_frame_equal(frames[key],expected)
        assert set(frames[key].user)==set(range(7))
    assert protocol['environment_resets']==7
    assert protocol['environment_advances']<protocol['independent_environment_advances']
    assert protocol['policy_frozen'] and len(protocol['policy_fingerprints'])==len(policies)
    assert protocol['score_parameters']==utils.score_parameters(j)
    for key in policies:assert protocol['member_score_parameters'][key]==utils.score_parameters(j)
    if domain=='tianchi':
        summary=utils.summarize(frames['zeroS'])
        assert summary['mean_orders'] is summary['mean_paid_minor'] is summary['mean_discount_minor'] is None
        assert frames['zeroS'].conversion_unit.eq('purchase_day').all()


class BatchActor(torch.nn.Module):
    def forward(self,states):
        index=torch.arange(len(states))%2
        return torch.stack((1-index,index),dim=1).float()


class BatchPolicy:
    def __init__(self):self.actor_inference=BatchActor()


@pytest.mark.parametrize('domain',('cosmetics','tianchi'))
def test_two_shards_preserve_every_user_batch_policy_counts_and_global_digests(domain):
    j=job(domain)
    policies={'zeroS':{'policy':PatternPolicy([0]),'normalizer':normalizer(),'cost':'S'},
        'lateD':{'policy':PatternPolicy([0,1,1]),'normalizer':normalizer(),'cost':'D'},
        'batchS':{'policy':BatchPolicy(),'normalizer':normalizer(),'cost':'S'}}
    whole,protocol=utils.evaluate_shared_policies(policies,ToyWeeklyEnv(domain=domain),j)
    parts=[utils.evaluate_shared_policies(policies,ToyWeeklyEnv(domain=domain),j,user_start=start,user_stop=stop)
        for start,stop in runner.evaluation_ranges(j['n_test'],j['eval_batch_size'])]
    merged,combined=runner.merge_shared_results(list(reversed(parts)),j)
    for key in whole:pd.testing.assert_frame_equal(merged[key],whole[key])
    for key in ('environment_resets','actual_warmup_resets','initial_state_cache_hits','environment_advances','episode_clones',
        'independent_environment_advances','transitions','initialization_attempts','initialization_attempts_by_user',
        'baseline_action_sequence_sha256','user_episode_seed_sha256','policy_fingerprints'):
        assert combined[key]==protocol[key]
    assert combined['evaluation_shards']==[[0,3],[3,7]]
    if domain=='tianchi':assert merged['batchS'][list(utils.UNOBSERVED_T)].isna().all().all()
    duplicated=copy.deepcopy(parts);duplicated[1][0]['batchS'].loc[0,'user']=6
    with pytest.raises(ValueError,match='duplicated'):runner.merge_shared_results(duplicated,j)
    with pytest.raises(ValueError,match='overlap'):runner.merge_shared_results([parts[0],parts[0]],j)
    with pytest.raises(ValueError,match='omit'):runner.merge_shared_results(parts[:1],j)
    bad=copy.deepcopy(parts);bad[1][1]['baseline_action_sequence_sha256']['Random']='tampered'
    with pytest.raises(ValueError,match='seeds'):runner.merge_shared_results(bad,j)


@pytest.mark.parametrize('domain',('cosmetics','tianchi'))
@pytest.mark.parametrize('coupon_cost',(100.,500.))
def test_shard_worker_disk_and_domain_report_match_whole_toy_evaluation(tmp_path,monkeypatch,domain,coupon_cost):
    j={**job(domain),'frozen_manifest':'toy-only','configs':{},'coupon_cost':coupon_cost}
    freeze={'manifest_sha256':'toy'}
    policies={'zeroS':{'policy':PatternPolicy([0]),'normalizer':normalizer(),'cost':'S'},
        'batchD':{'policy':BatchPolicy(),'normalizer':normalizer(),'cost':'D'}}
    selected={}
    for key,member in policies.items():
        directory=tmp_path/key;directory.mkdir()
        with (directory/'checkpoint_final.pkl').open('wb') as stream:pickle.dump(member['policy'],stream)
        (directory/'normalizer.json').write_text(json.dumps(member['normalizer']))
        summary={'job':{**j,'key':key,'cost':member['cost']},'frozen_world':freeze,'learning':{'optimizer_steps':0},
            'checkpoint_sha256':utils.sha256(directory/'checkpoint_final.pkl'),
            'normalizer_sha256':utils.sha256(directory/'normalizer.json')}
        (directory/'training_summary.json').write_text(json.dumps(summary));selected[key]=summary
    (tmp_path/'matrix_summary.json').write_text(json.dumps({'jobs':{k:s['job'] for k,s in selected.items()},
        'jobs_trained':{key:{} for key in selected}}))
    monkeypatch.setattr(runner,'validate_freeze',lambda *args:freeze)
    monkeypatch.setattr(runner,'make_env',lambda job,weeks,cost=None:ToyWeeklyEnv(domain=domain,cost=cost or 'S',coupon_cost=job['coupon_cost']))
    shards=[runner.evaluate_domain_shard(str(tmp_path),domain,start,stop)
        for start,stop in runner.evaluation_ranges(j['n_test'],j['eval_batch_size'])]
    completed=runner.evaluate_domain(str(tmp_path),domain,list(reversed(shards)))
    expected,protocol=utils.evaluate_shared_policies(policies,ToyWeeklyEnv(domain=domain,coupon_cost=coupon_cost),j)
    assert set(completed)==set(policies)
    for key in policies:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path/key/'evaluation_users.parquet'),expected[key])
        assert completed[key]['evaluation']==utils.summarize(expected[key])
    block=json.loads((tmp_path/'shared_evaluation'/domain/'summary.json').read_text())
    assert block['evaluation_protocol']['baseline_action_sequence_sha256']==protocol['baseline_action_sequence_sha256']
    assert block['evaluation_protocol']['users']==7 and len(block['evaluation_protocol']['shard_protocols'])==2
    for name in ('Never','Always','Random'):
        for cost,field in [('S','sends'),('D','redemptions')]:
            values=1000*expected[name].converted_sessions-coupon_cost*expected[name][field]
            assert block['fixed_policy_baselines'][name][cost]['mean_return']==float(values.mean())
    changed=Path(shards[0]['directory'])/'zeroS.parquet';changed.write_bytes(changed.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='rows changed'):runner.evaluate_domain(str(tmp_path),domain,shards)


@pytest.mark.parametrize('name',list(AGENTS))
@pytest.mark.parametrize('coupon_cost',(100.,500.))
def test_toy_training_records_real_steps_checkpoint_and_final_ppo_tail_curve(tmp_path,name,coupon_cost):
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        kwargs={'seed':3}
        if name in {'PPO','SAC','TD3'}:kwargs['device']='cpu'
        if name in {'SAC','TD3'}:kwargs.update(learning_starts=4,batch_size=4,train_interval=1)
        policy=AGENTS[name](39,**kwargs);directory=tmp_path/name;directory.mkdir()
        frame,learning=utils.train_policy(policy,name,ToyWeeklyEnv(coupon_cost=coupon_cost),normalizer(),{**job(),'coupon_cost':coupon_cost},directory)
        assert len(frame)==2 and learning['transitions']==6
        np.testing.assert_array_equal(frame.reward_score,1000*frame.converted_sessions-coupon_cost*frame.sends)
        assert (directory/'checkpoint_final.pkl').exists()
        with (directory/'checkpoint_final.pkl').open('rb') as stream:restored=pickle.load(stream)
        assert utils.policy_fingerprint(policy)==utils.policy_fingerprint(restored)
        curve=pd.read_csv(directory/'training_curve.csv')
        assert curve.iloc[-1].steps==6 and curve.iloc[-1].users==2
        assert curve.iloc[-1].optimizer_steps==learning['optimizer_steps']
        if name=='PPO':assert learning['ppo_learn_calls']==2 and learning['optimizer_steps']==8
        if name in {'SAC','TD3'}:assert learning['optimizer_steps']>0
    finally:torch.set_num_threads(previous)


def test_tianchi_training_and_normalization_preserve_null_raw_outcomes(tmp_path):
    env=ToyWeeklyEnv(domain='tianchi')
    calibrated=utils.fit_calibration(env,2,3,'tianchi',11,tmp_path/'cache')
    assert calibrated['calibration_transitions']==6 and len(calibrated['features'])==39
    assert np.isfinite(utils.scale_states(np.full(39,np.nan),calibrated)).all()
    directory=tmp_path/'train';directory.mkdir()
    frame,_=utils.train_policy(AGENTS['UCB'](39), 'UCB',env,calibrated,job('tianchi'),directory)
    assert frame.orders.isna().all() and frame.paid_minor.isna().all()
    assert utils.summarize(frame)['mean_paid_minor'] is None


def test_normalizer_cache_reuses_only_identical_calibration_episode_bank(tmp_path):
    first=ToyWeeklyEnv();before=ToyWeeklyEnv.reset_calls
    one=utils.fit_calibration(first,3,3,'cosmetics',17,tmp_path)
    assert ToyWeeklyEnv.reset_calls-before==3
    two=utils.fit_calibration(ToyWeeklyEnv(cost='D'),3,3,'cosmetics',17,tmp_path)
    assert ToyWeeklyEnv.reset_calls-before==3 and one==two


def test_changed_cache_bytes_are_rejected_before_state_restore(tmp_path):
    utils.cache_reset(ToyWeeklyEnv(),19,tmp_path)
    path=next(tmp_path.glob('*.pkl'));path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):utils.cache_reset(ToyWeeklyEnv(),19,tmp_path)


# Existing native fixtures use tiny fitted/source models, never the real world assets.
from test_v4_main_campaign import factory as main_factory
from test_v4_tianchi_campaign import factory as daily_factory
from test_v4_tianchi_daily import source,model
from test_v4_canonical_response import fixture_model
from simulate.v4.simulator import main_campaign as native_main
from simulate.v4.simulator.canonical_response import CanonicalResponseAdapter
from simulate.v4.simulator.canonical_campaign import CanonicalTianchiDailyCampaign


def test_native_pending_main_cache_keeps_cart_history_and_active_row_alias(main_factory,monkeypatch,tmp_path):
    def policy(model,context,component,probabilities=None):
        action=('cart','purchase','end')[min(2,context['session_decision_count'])]
        return {a:float(a==action) for a in native_main.shopping.ACTIONS}
    monkeypatch.setattr(native_main.shopping,'policy',policy)
    monkeypatch.setattr(native_main,'sample_session_offsets',lambda *args:([8*86400.],'fixture'))
    template=main_factory(warmup_days=14,horizon_days=14,
        edit=lambda model:model['profiles'][0].update(choice_gap_seconds=[20*86400.]))
    uncached=copy.copy(template);uncached.reset(51)
    saved=copy.copy(template);utils.cache_reset(saved,51,tmp_path/'cache')
    restored=copy.copy(template);utils.cache_reset(restored,51,tmp_path/'cache')
    assert restored._active['row'] is restored.records['sessions'][-1]
    assert restored._active['diagnostic'] is restored.records['session_diagnostics'][-1]
    assert restored._active['planned']==uncached._active['planned']
    assert restored.state['cart']==uncached.state['cart']
    for env in (uncached,saved,restored):env.advance(0,14)
    for name in ('sessions','events','orders','order_items','session_diagnostics'):
        pd.testing.assert_frame_equal(uncached.tables[name],restored.tables[name])
        pd.testing.assert_frame_equal(saved.tables[name],restored.tables[name])


def test_native_canonical_daily_cache_keeps_type_latent_and_day_rng_alias(daily_factory,tmp_path):
    native=daily_factory(horizon_days=14)
    spec,bank,_=fixture_model('tianchi_daily',native.model_shas);spec['default_strength']=0.
    adapter=CanonicalResponseAdapter(spec,bank)
    template=CanonicalTianchiDailyCampaign(native.config,canonical_adapter=adapter,warmup_days=30,min_history_days=0,horizon_days=14)
    direct=copy.copy(template);direct.reset(52)
    saved=copy.copy(template);utils.cache_reset(saved,52,tmp_path/'cache')
    restored=copy.copy(template);utils.cache_reset(restored,52,tmp_path/'cache')
    assert restored.canonical_adapter is template.canonical_adapter
    assert restored.response is not saved.response and not np.shares_memory(restored.response['latent48'],saved.response['latent48'])
    assert restored.response['canonical_type']==direct.response['canonical_type']
    assert restored._daily.rng is restored.day_rng and restored.day_rng is not saved.day_rng
    for env in (direct,saved,restored):env.advance(1,7);env.advance(0,7)
    for name in ('raw_daily','runtime_lifecycle','coupon_redemptions','coupons'):
        pd.testing.assert_frame_equal(direct.tables[name],restored.tables[name])
        pd.testing.assert_frame_equal(saved.tables[name],restored.tables[name])


def test_runner_constructs_real_daily_factory_without_a_retail_domain_attribute(daily_factory,tmp_path):
    native=daily_factory(warmup_days=364,horizon_days=91)
    spec,bank,_=fixture_model('tianchi_daily',native.model_shas)
    spec['new_world_calibrated']=True
    bank_path=tmp_path/'runner_bank.parquet';bank.to_parquet(bank_path,index=False)
    spec.update(latent_bank_file=str(bank_path),latent_bank_sha256=utils.sha256(bank_path))
    model_path=tmp_path/'runner_canonical.json';model_path.write_text(json.dumps(spec))
    config=copy.deepcopy(native.config);config['source_simulation']['canonical_response_model_path']=str(model_path)
    env=runner.make_env({'config':config,'domain':'tianchi','cost':'S','freeze':{'manifest_sha256':'toy'}},13)
    assert isinstance(env,CanonicalTianchiDailyCampaign) and not hasattr(env,'domain')
    assert env.canonical_adapter._model['domain']=='tianchi'
    assert env._benchmark_freeze_sha256=='toy' and env.horizon_days==91
    state=env.reset(42)
    assert state.shape==(39,) and env.now==env.decision_start
    next_state,reward,done,info=env.advance(0,7)
    assert next_state.shape==(39,) and not done and np.isfinite(reward)
    assert info['orders'] is info['paid_minor'] is info['discount_minor'] is None


def test_runner_rejects_foreign_worktree_imports_even_with_matching_module_names(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import sys
    monkeypatch.setitem(sys.modules,'simulate.v4.foreign_runtime_fixture',SimpleNamespace(__file__=str(tmp_path/'old_worktree.py')))
    with pytest.raises(ValueError,match='outside the active checkout'):runner.assert_active_checkout()
