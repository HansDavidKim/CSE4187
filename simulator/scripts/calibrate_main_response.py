"""Fresh-world canonical strength calibration and daily source-IRL primary check."""
from __future__ import annotations

import argparse,ast,copy,hashlib,json,pickle,time,traceback
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from simulate.v4.configuration import load_config
from simulate.v4.irl.source_contract import file_sha
from simulate.v4.simulator.canonical_campaign import make_campaign_env
from simulate.v4.simulator.canonical_response import response_offsets
from simulate.v4.simulator.coupon_response import sample_coupon_response
from simulate.v4.simulator.tianchi_campaign import TianchiDailyCampaignEnv


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def legacy_search(path):
    # Reuse the exact two original finite-MC search helpers without importing old-world runtimes.
    nodes=[n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('select_trial','next_strength')]
    if len(nodes)!=2:raise ValueError('original finite-MC response helpers changed')
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['select_trial'],namespace['next_strength']


def episode_seed(phase,role,seed,user):
    key=json.dumps(['weekly-coupon-v1',phase,role,int(seed),int(user)]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big')


def fork(initial):
    memo={}
    # Native product catalog is immutable and can be large. Runtime records/state remain one copied graph.
    if 'products' in initial.records:memo[id(initial.records['products'])]=initial.records['products']
    return copy.deepcopy(initial,memo)


def set_strength(env,value):
    if hasattr(env,'canonical_adapter'):env.canonical_adapter.set_strength(env,value)
    else:env.source_coupon_strength=float(value)


def purchases(env,start,days):
    upper=start+pd.Timedelta(days=days)
    if 'raw_daily' in env.records:return any(r['purchase'] and start<=r['day_start']<upper for r in env.records['raw_daily'])
    return any(r['event_type']=='purchase' and start<=r['event_at']<upper for r in env.records['events'])


def active(env,start,days):
    upper=start+pd.Timedelta(days=days)
    if 'raw_daily' in env.records:return any((r['view'] or r['purchase']) and start<=r['day_start']<upper for r in env.records['raw_daily'])
    return any(start<=r['started_at']<upper for r in env.records['sessions'])


def shopping_path(env):
    if 'raw_daily' in env.records:return [(r['day_start'],r['view'],r['purchase'],r['outcome']) for r in env.records['raw_daily']]
    return [(r['event_at'],r['event_type'],r['product_id'],r['quantity']) for r in env.records['events']
        if r['event_type'] in {'view','cart','remove_from_cart','purchase'}]


def future_rows(env,start):
    """Full future daily rows or linked canonical retail raw outcome relations, without duplicated catalogs."""
    if 'raw_daily' in env.records:
        result={'raw_daily':[r for r in env.records['raw_daily'] if r['day_start']>=start],
            'coupons':list(env.records['coupons']),'coupon_redemptions':list(env.records['coupon_redemptions'])}
    else:
        events=[r for r in env.records['events'] if r['event_at']>=start]
        event_ids={r['event_id'] for r in events};session_ids={r['session_id'] for r in events}
        orders=[r for r in env.records['orders'] if r['purchase_event_id'] in event_ids]
        order_ids={r['order_id'] for r in orders}
        result={'events':events,'sessions':[r for r in env.records['sessions'] if r['started_at']>=start or r['session_id'] in session_ids],
            'orders':orders,'order_items':[r for r in env.records['order_items'] if r['order_id'] in order_ids],
            'coupons':list(env.records['coupons'])}
    result['runtime_lifecycle']=[r for r in env.records['runtime_lifecycle'] if r['as_of_at']>=start]
    return result


def cohort(template,role,seed,users,strength,phase,*,invariance=False,initial_states=None,raw_prefix=None):
    rows=[];raw={};began=time.monotonic()
    for user in range(users):
        customer_seed=episode_seed(phase,role,seed,user)
        if initial_states is not None and user<len(initial_states):initial=initial_states[user]
        else:
            initial=fork(template) if hasattr(template,'records') else copy.deepcopy(template)
            initial.reset(customer_seed)
            if initial_states is not None:initial_states.append(initial)
        initial_digest=hashlib.sha256(pickle.dumps((initial.state,initial.theta,initial.rng.bit_generator.state),protocol=5)).hexdigest()
        baseline,treated=fork(initial),fork(initial)
        start=initial.decision_start
        for env,action in ((baseline,0),(treated,1)):
            set_strength(env,strength);env.advance(action,template.horizon_days)
            np.testing.assert_array_equal(env.theta,initial.theta)
            if hasattr(initial,'canonical_adapter'):
                assert env.response['canonical_type']==initial.response['canonical_type']
                np.testing.assert_array_equal(env.response['latent48'],initial.response['latent48'])
        if invariance:
            if shopping_path(baseline)!=shopping_path(treated):raise ValueError('zero strength changed source shopping path')
            if 'sessions' in baseline.records and baseline.records['sessions']!=treated.records['sessions']:
                raise ValueError('zero strength changed native START/END paths')
            for name in ('rng','visit_rng','review_rng','day_rng','entry_rng','response_rng'):
                if hasattr(baseline,name) and getattr(baseline,name).bit_generator.state!=getattr(treated,name).bit_generator.state:
                    raise ValueError('zero strength consumed a source RNG: '+name)
        if hasattr(initial,'canonical_adapter') and initial.response['persona']!=role:
            raise ValueError('calibration must condition the original joint prior on its frozen response role')
        rows.append({'user':user,'persona':role,'seed':customer_seed,'source_profile_id':initial.source_profile_id,
            'source_actor':int(initial.actor if hasattr(initial,'actor') else initial.component),
            'canonical_type':initial.response['canonical_type'] if hasattr(initial,'canonical_adapter') else None,
            'initial_digest':initial_digest,'quiet_prior14d':not active(initial,start-pd.Timedelta(days=14),14),
            'p0':int(purchases(baseline,start,14)),'p1':int(purchases(treated,start,14)),
            'purchase_4w_control':int(purchases(baseline,start,28)) if template.horizon_days>=28 else None,
            'purchase_4w_receipt':int(purchases(treated,start,28)) if template.horizon_days>=28 else None,
            'return0':int(active(baseline,start,7)),'return1':int(active(treated,start,7)),
            'static_gmm_draw_persistent':True,'same_initial_prefix':True})
        if raw_prefix is not None:
            for branch,env in (('control',baseline),('receipt',treated)):
                for name,records in future_rows(env,start).items():
                    raw.setdefault(name,[]).extend({'pair_user':user,'pair_seed':customer_seed,'pair_role':role,'pair_branch':branch,**r} for r in records)
        if (user+1)%128==0:print(json.dumps({'stage':phase,'role':role,'strength':strength,'users':user+1,'seconds':time.monotonic()-began}),flush=True)
    frame=pd.DataFrame(rows)
    if raw_prefix is not None:
        raw_prefix=Path(raw_prefix);raw_prefix.mkdir(exist_ok=False)
        for name,records in raw.items():
            if records:pd.DataFrame(records).to_parquet(raw_prefix/f'{name}.parquet',index=False)
    difference=frame.p1.to_numpy(float)-frame.p0.to_numpy(float)
    quiet=frame.loc[frame.quiet_prior14d];qdiff=quiet.return1.to_numpy(float)-quiet.return0.to_numpy(float)
    summary={'strength':strength,'users':users,'p0':float(frame.p0.mean()),'p1':float(frame.p1.mean()),
        'difference':float(difference.mean()),'paired_se':float(difference.std(ddof=1)/np.sqrt(users)),
        'return0':float(frame.return0.mean()),'return1':float(frame.return1.mean()),'quiet14_users':len(quiet),
        'quiet14_return0':float(quiet.return0.mean()) if len(quiet) else None,'quiet14_return1':float(quiet.return1.mean()) if len(quiet) else None,
        'quiet14_return_paired_se':float(qdiff.std(ddof=1)/np.sqrt(len(quiet))) if len(quiet)>1 else None,
        'seconds':time.monotonic()-began,'static_gmm_draw_persistent':True,'same_prefix_for_both_branches':True}
    return frame,summary


class SourceDailyCoupon:
    def __init__(self,model,receipt):self.model,self.receipt=model,receipt
    def __deepcopy__(self,memo):return self
    def response_sampler(self,env,at):
        known=[r for r in env.records['raw_daily'] if r['purchase'] and r['known_at']<=at]
        last=max((r['day_start'] for r in known),default=None)
        result=sample_coupon_response(self.receipt,env.seed,env.uid,at,last)
        result['response_selected']=bool(result['response_selected'])
        if not result['response_selected']:result['response_ready_at']=None
        return result
    def runtime_reward_offsets(self,env,at):
        age=[(at-r['sent_at']).total_seconds()/86400 for r in env._receipts if r['sent_at']<=at]
        purchase,returned=self.model['coefficients']
        shifted=response_offsets({'purchase_offset':purchase,'return_offset':returned,'strength':getattr(env,'source_coupon_strength',1.)},age)
        return {'purchase_reward_shift':shifted['purchase'],'return_reward_shift':shifted['return']}


def lock_inputs(config,extra=()):
    paths=[Path(p) for p in extra]
    for key,value in config['source_simulation'].items():
        if key.endswith('_path') and value:paths.append(Path(value))
        if key=='daily_actor_model_paths':paths.extend(Path(p) for p in value.values())
    source=Path(__file__).resolve().parents[1]/'src/simulate/v4'
    names=['simulator/'+x for x in ('main_campaign','tianchi_campaign','canonical_campaign','canonical_response','canonical_actor_mapping',
        'private_return','online_features','population_profiles','population_timing','coupon_response','raw_store','item_choice','review_events')]
    names+=['irl/'+x for x in ('tianchi_daily','fit_latent_shopping','shopping_history','fit_return','fit_joint_latent_shopping',
        'reward_gmm','choice_gmm','conditional_return')]+['time_spec','features']
    if config['source_simulation'].get('source_history_schedule_path'):names.append('simulator/source_history')
    paths.extend(source/(name+'.py') for name in names)
    paths.append(Path(__file__))
    canonical=config['source_simulation'].get('canonical_response_model_path')
    if canonical:
        m=json.loads(Path(canonical).read_text());paths.append(Path(canonical).parent/m['latent_bank_file'])
    return {str(p.resolve()):file_sha(p) for p in paths}


def verify(locks):
    changed=[p for p,sha in locks.items() if file_sha(p)!=sha]
    if changed:raise RuntimeError('frozen calibration world changed: '+str(changed))


def calibrate(args):
    config=load_config(args.config);source=Path(config['source_simulation']['canonical_response_model_path'])
    model=json.loads(source.read_text());output=args.output;output.mkdir(parents=True,exist_ok=False)
    original=copy.deepcopy(model);began=time.monotonic()
    active=Path(__file__).resolve().parents[1]
    helper=active/'../../v6-paper/simulator/scripts/calibrate_v6_paper_response.py'
    select_trial,next_strength=legacy_search(helper)
    locks=lock_inputs(config,[args.config,helper]);write(output/'protocol.json',{'input_sha256':locks,
        'domain':args.domain,'calibration_users_per_persona':args.calibration_users,'validation_users_per_persona':args.validation_users,
        'calibration_seed':args.calibration_seed,'validation_seed':args.validation_seed,'maximum_iterations':8,
        'strength_bounds':[.01,128.],'gain_tolerance':.02,'selection':'original finite-MC bracket helper, minimum absolute14day purchase gain error',
        'zero_pairs_per_persona':20,'warmup_days':364,'minimum_history_days':91,'fitted_parameters':'one strength per supported persona',
        'fixed_parameters':'original49x4 bytes, four-type probabilities, purchase:return ratio, sixweek kernel and source world',
        'role_conditioning':'original joint P(G,K|R) proportional to P(G)P(K|G)P(R|K); original static GMM conditional onK',
        'validation_used_for_selection':False,'return7_used_for_selection':False,'old_world_sourceIRL_evidence_reused':False})
    reports={};selected={}
    for role,spec in sorted(model['personas'].items()):
        target=spec['purchase_14d_receipt']-spec['purchase_14d_control']
        env=make_campaign_env(config,domain=args.domain,calibration_persona=role,require_calibrated=False,
            warmup_days=364,min_history_days=91,horizon_days=14,feature_builder=lambda _:np.zeros(39))
        initial=[]
        zero,zsummary=cohort(env,role,args.calibration_seed,20,0.,'canonical-calibration',invariance=True,
            initial_states=initial,raw_prefix=output/f'{role}_zero_raw')
        zero.to_parquet(output/f'{role}_zero.parquet',index=False)
        trials=[];best=None;strength=model.get('strength_by_persona',{}).get(role,model['default_strength'])
        for iteration in range(8):
            frame,summary=cohort(env,role,args.calibration_seed,args.calibration_users,strength,'canonical-calibration',
                initial_states=initial,raw_prefix=output/f'{role}_trial{iteration}_raw')
            summary.update(iteration=iteration,target=target,error=summary['difference']-target)
            trials.append(summary)
            if select_trial(trials,target) is summary:best=frame
            write(output/f'{role}_trial{iteration}.json',summary)
            print(json.dumps({'domain':args.domain,'role':role,**summary}),flush=True)
            if abs(summary['error'])<=.02:break
            proposed=next_strength(trials,target)
            if any(np.isclose(proposed,t['strength']) for t in trials):break
            strength=proposed
        chosen=select_trial(trials,target);selected[role]=chosen['strength']
        best.to_parquet(output/f'{role}_calibration.parquet',index=False)
        reports[role]={'target_purchase14_gain':target,'selected':chosen,'trials':trials,'zero_strength_exact':True,
            'zero_summary':zsummary,'validation_used_for_selection':False,'return7_used_for_selection':False}
        del initial
    # Strength selection for every persona is immutable before any new validation outcome is read.
    write(output/'selected_before_validation.json',{'strength_by_persona':selected,'selection':'calibration only'})
    for role,spec in sorted(model['personas'].items()):
        env=make_campaign_env(config,domain=args.domain,calibration_persona=role,require_calibrated=False,
            warmup_days=364,min_history_days=91,horizon_days=28,feature_builder=lambda _:np.zeros(39))
        frame,summary=cohort(env,role,args.validation_seed,args.validation_users,selected[role],'canonical-validation',
            raw_prefix=output/f'{role}_validation_raw')
        frame.to_parquet(output/f'{role}_validation.parquet',index=False)
        summary['purchase_gain_ci95']=[summary['difference']+sign*1.96*summary['paired_se'] for sign in (-1,1)]
        summary['purchase_4w_control']=float(frame.purchase_4w_control.mean());summary['purchase_4w_receipt']=float(frame.purchase_4w_receipt.mean())
        reports[role]['validation']=summary
        write(output/f'{role}_result.json',reports[role])
    verify(locks)
    completed=all(abs(row['selected']['error'])<=.02 for row in reports.values())
    model['strength_by_persona']=selected;model['new_world_calibrated']=completed
    model['strength_status']='fresh_world_purchase_gain_calibrated' if completed else 'fresh_world_purchase_gain_calibration_incomplete'
    model['fresh_world_calibration']={'reports':reports,'input_sha256':locks,'seconds':time.monotonic()-began,
        'validation_used_for_selection':False,'return7_used_for_selection':False,'source_model_sha256':file_sha(source)}
    for key in ('canonical_assignment','personas','original_matrix_json_base64','original_matrix_float64_sha256'):
        assert model[key]==original[key]
    # Keep the bank in place by recording its absolute immutable path.
    model['latent_bank_file']=str((source.parent/model['latent_bank_file']).resolve())
    write(output/'model.json',model)
    config['source_simulation']['canonical_response_model_path']=str((output/'model.json').resolve())
    write(output/'config.json',config)
    write(output/'verification.json',{'domain':args.domain,'model_sha256':file_sha(output/'model.json'),
        'all_zero_pairs_exact':True,'all_purchase_calibration_errors_le_0_02':completed,
        'all_purchase_validation_errors_le_0_02':all(abs(r['validation']['difference']-r['target_purchase14_gain'])<=.02 for r in reports.values()),
        'selected_strengths':selected,'changed_inputs':[],'seconds':time.monotonic()-began})
    print(json.dumps({'stage':'complete','domain':args.domain,'completed':completed,'selected':selected,'seconds':time.monotonic()-began}),flush=True)


def primary(args):
    config=load_config(args.config);output=args.output;output.mkdir(parents=True,exist_ok=False)
    source=Path(args.source_model);model=json.loads(source.read_text());receipt_path=Path(args.receipt_model)
    if not model['converged']:raise ValueError('fresh source reward fit must converge before the primary check')
    callback=SourceDailyCoupon(model,json.loads(receipt_path.read_text()))
    env=TianchiDailyCampaignEnv(config,warmup_days=364,min_history_days=91,horizon_days=28,
        response_sampler=callback.response_sampler,runtime_reward_offsets=callback.runtime_reward_offsets,feature_builder=lambda _:np.zeros(39))
    if model['daily_actor_sha256']!=env.actor_sha:raise ValueError('fresh source augmentation refers to another daily world')
    locks=lock_inputs(config,[args.config,source,receipt_path])
    target=.09089999999999998
    write(output/'protocol.json',{'input_sha256':locks,'users':1000,'seed':args.calibration_seed,'zero_pairs':20,
        'one_contact_at_t0':True,'original_minimum_positive_purchase14_gain':target,
        'criterion':'original target vs upper95percent paired MC interval, no coefficient calibration to this check',
        'source_reward_units':'daily purchase/activity, not30d GLM or choice logit','old_world_result_reused':False})
    _,zero=cohort(env,'all_source_personas',args.calibration_seed,20,0.,'source-primary',invariance=True,raw_prefix=output/'zero_raw')
    frame,summary=cohort(env,'all_source_personas',args.calibration_seed,1000,1.,'source-primary',raw_prefix=output/'primary_raw')
    frame.to_parquet(output/'pairs.parquet',index=False);verify(locks)
    summary.update(zero_shift=zero,zero_shift_invariant=True,original_minimum_positive_purchase_gain=target,
        purchase_gain_ci95=[summary['difference']+sign*1.96*summary['paired_se'] for sign in (-1,1)],
        old_world_result_reused=False,source_world='frozen weekly daily6IRL+GMM+entry',causal_effect_identified=False)
    summary['replicates_original_positive_gain']=summary['purchase_gain_ci95'][1]>=target
    write(output/'result.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('canonical','source-primary'),required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',choices=('cosmetics','tianchi'));p.add_argument('--source-model');p.add_argument('--receipt-model')
    p.add_argument('--calibration-users',type=int,default=256);p.add_argument('--validation-users',type=int,default=512)
    p.add_argument('--calibration-seed',type=int,default=20261125);p.add_argument('--validation-seed',type=int,default=20261126)
    args=p.parse_args()
    if args.calibration_users<20 or args.validation_users<2:p.error('at least20 calibration and2 validation pairs required')
    with threadpool_limits(limits=1):
        try:(calibrate if args.mode=='canonical' else primary)(args)
        except Exception as error:
            if args.output.exists() and not (args.output/'failure.json').exists():write(args.output/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
            raise


if __name__=='__main__':main()
