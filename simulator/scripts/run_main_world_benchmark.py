"""Seven policies × three native sources × S/D, gated on one frozen world.

Ported weekly training, normalization and identical-action-prefix evaluation use
only this checkout's modules. --plan prepares a reviewable42-cell budget without
source rollout or training. Real learning requires every source/coupon/104w gate
and the exact SHA freeze. Trusted pickle checkpoints/caches stay local.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import importlib
import json
import multiprocessing
from pathlib import Path
import pickle
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from models.agents import AGENTS
from simulate.v4.configuration import load_config
from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.time_spec import weekly_metadata
from scripts.main_world_benchmark_utils import (episode_seed,fit_calibration,gradient_counts,
    train_policy,evaluate_shared_policies,summarize,sha256,write_json,protocol_digest,score_parameters)

ROOT=Path(__file__).resolve().parents[1]
DOMAINS=('cosmetics','electronics','tianchi')
GATES=('source','coupon','export_104w')


def assert_active_checkout():
    for name,module in tuple(sys.modules.items()):
        if name.startswith('simulate.v4.') or name=='models.agents' or name=='scripts.main_world_benchmark_utils':
            filename=getattr(module,'__file__',None)
            if filename is None:continue
            path=Path(filename).resolve()
            expected=ROOT/'scripts' if name.startswith('scripts.') else ROOT/'src'
            if not path.is_relative_to(expected):
                raise ValueError(f'benchmark imported mutable code outside the active checkout: {name}: {path}')


def _path(value,base):
    path=Path(value).expanduser()
    return (path if path.is_absolute() else base/path).resolve()


def required_files(config_path):
    """Only runtime inputs: native config/models, canonical model+48D bank, code."""
    config_path=Path(config_path).resolve();config=load_config(config_path)
    files={config_path,ROOT/'src/models/agents.py',Path(__file__).resolve(),
        ROOT/'scripts/main_world_benchmark_utils.py',*(ROOT/'src/simulate/v4').rglob('*.py')}
    settings=config['source_simulation']
    for key,value in settings.items():
        if key.endswith('_path') and value:files.add(_path(value,config_path.parent))
        elif key.endswith('_paths') and isinstance(value,dict):files.update(_path(v,config_path.parent) for v in value.values())
    canonical=settings.get('canonical_response_model_path')
    if not canonical:raise ValueError('policy configs require their separately calibrated canonical_response_model_path')
    canonical_path=_path(canonical,config_path.parent)
    artifact=json.loads(canonical_path.read_text())
    files.add(_path(artifact['latent_bank_file'],canonical_path.parent))
    return {p.resolve() for p in files}


def validate_freeze(manifest_path,configs):
    """No numeric quality criterion is invented here; signed-off gates are bound."""
    if manifest_path is None:raise ValueError('learning/evaluation requires a completed --frozen-manifest')
    assert_active_checkout()
    manifest_path=Path(manifest_path).resolve();manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema_version')!=1 or manifest.get('status')!='frozen':raise ValueError('world freeze is incomplete')
    stamp=datetime.fromisoformat(manifest.get('frozen_at',''))
    if stamp.tzinfo is None or stamp>datetime.now(timezone.utc):raise ValueError('world freeze needs a completed aware timestamp')
    hashes=manifest.get('file_sha256',{})
    if not isinstance(hashes,dict) or not hashes:raise ValueError('world freeze needs exact file SHA256 bindings')
    frozen_configs=manifest.get('configs',{})
    if set(frozen_configs)!=set(DOMAINS):raise ValueError('freeze must name the three adopted domain configs')
    if any(_path(path,manifest_path.parent)!=_path(frozen_configs.get(domain,''),manifest_path.parent) for domain,path in configs.items()):
        raise ValueError('requested source config differs from the adopted frozen world')
    required=set()
    for name in GATES:
        gate=manifest.get('gates',{}).get(name,{})
        if gate.get('status')!='complete' or gate.get('passed') is not True or set(gate.get('domains',[]))!=set(DOMAINS):
            raise ValueError(f'{name} gate must complete for all three domains before policy learning')
        reports=gate.get('reports',[])
        if not reports:raise ValueError(f'{name} gate requires bound evidence reports')
        required.update(_path(path,manifest_path.parent) for path in reports)
    for domain,path in configs.items():
        if domain not in DOMAINS:raise ValueError('unknown native source domain')
        required.update(required_files(path))
    if not {str(p) for p in required}<=set(hashes):raise ValueError('world freeze omits runtime, source/coupon or104w evidence inputs')
    for path,expected in hashes.items():
        if sha256(path)!=expected:raise ValueError(f'frozen world input changed: {path}')
    return {'manifest_path':str(manifest_path),'manifest_sha256':sha256(manifest_path),
        'frozen_at':manifest['frozen_at'],'gates':{key:manifest['gates'][key] for key in GATES}}


def make_env(job,weeks,cost=None):
    from simulate.v4.simulator.canonical_campaign import make_campaign_env
    assert_active_checkout()
    env=make_campaign_env(job['config'],domain=job['domain'],cost_model=cost or job['cost'],
        require_calibrated=True,horizon_days=weeks*7,
        **{'score_'+key:value for key,value in score_parameters(job).items()})
    assert_active_checkout()
    if env.warmup_days!=364 or env.coupon_valid_days!=28 or env.canonical_adapter._model['domain']!=job['domain']:
        raise ValueError('policy worlds require exact52w warmup,28d coupons and their native domain')
    env._benchmark_freeze_sha256=job['freeze']['manifest_sha256']
    return env


def _device(device):
    if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS requested but unavailable; no silent device fallback')
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA requested but unavailable')


def train_job(job):
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        checked=validate_freeze(job['frozen_manifest'],job['configs'])
        if checked!=job['freeze']:raise ValueError('world freeze differs from the preassigned matrix')
        _device(job['device'])
        output=Path(job['output']);output.mkdir(parents=True,exist_ok=False)
        env=make_env(job,job['train_weeks'])
        normalizer=fit_calibration(env,job['calibration_users'],job['train_weeks'],job['domain'],
            job['calibration_seed'],job['initial_state_cache'])
        write_json(output/'normalizer.json',normalizer)
        kwargs={'seed':job['seed']}
        if job['agent'] in {'PPO','SAC','TD3'}:kwargs['device']=job['device']
        if job['agent'] in {'SAC','TD3'}:kwargs.update(learning_starts=job['learning_starts'],train_interval=1,batch_size=job['sac_batch_size'])
        policy=AGENTS[job['agent']](39,**kwargs)
        started=datetime.now(timezone.utc).isoformat()
        frame,learning=train_policy(policy,job['agent'],env,normalizer,job,output)
        if len(frame)!=job['n_train'] or learning['transitions']!=job['n_train']*job['train_weeks']:
            raise ValueError('training omitted a scheduled customer or transition')
        if validate_freeze(job['frozen_manifest'],job['configs'])!=checked:raise ValueError('world changed during learning')
        summary={'stage':'trained','job':job,'training_started_at':started,'training':summarize(frame),'learning':learning,
            'feature_names':list(EXPANDED_FEATURES),'feature_count':39,'model_sha256':env.model_shas,
            'runtime_versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__},
            'config_sha256':sha256(job['config']),'checkpoint_sha256':sha256(output/'checkpoint_final.pkl'),
            'normalizer_sha256':sha256(output/'normalizer.json'),'frozen_world':checked,
            'calendar':{'warmup_days':364,'training_weeks':job['train_weeks'],'evaluation_weeks':job['test_weeks'],'coupon_valid_days':28},
            'independence':'separate hashed calibration/training/evaluation episode seeds; same frozen source population/models reused; not independent real-user forecasting',
            'outcome_observability':'Tianchi converted_sessions means purchase-presence days; orders/payments/discounts remain null' if job['domain']=='tianchi' else 'source-specific retail sessions and raw order/payment amounts',
            'scope':'actual optimizer updates and transitions only; no extrapolated curve; annual reward undiscounted',**weekly_metadata()}
        write_json(output/'training_summary.json',summary)
        write_json(output/'progress.json',{'stage':'trained','training_steps':learning['transitions'],'evaluation_status':'pending',**gradient_counts(policy,learning['ppo_learn_calls'])})
        return summary


def _domain_training(root,domain):
    matrix=json.loads((root/'matrix_summary.json').read_text())
    selected={key:json.loads((root/key/'training_summary.json').read_text()) for key in matrix['jobs_trained'] if matrix['jobs'][key]['domain']==domain}
    expected={key for key,job in matrix['jobs'].items() if job['domain']==domain}
    if set(selected)!=expected:raise ValueError('shared evaluation requires all scheduled trained domain policies')
    first=next(iter(selected.values()));job=first['job']
    if score_parameters(job)!=score_parameters(matrix.get('protocol',job)):
        raise ValueError('training matrix score weights differ from its domain jobs')
    checked=validate_freeze(job['frozen_manifest'],job['configs'])
    for key,summary in selected.items():
        folder=root/key
        if (summary['frozen_world']!=checked or summary['job']['n_test']!=job['n_test'] or summary['job']['test_weeks']!=job['test_weeks']
                or summary['job']['evaluation_seed']!=job['evaluation_seed']
                or summary['job']['eval_batch_size']!=job['eval_batch_size']
                or score_parameters(summary['job'])!=score_parameters(job)):raise ValueError('policy pairing/freeze/score weights differ')
        if sha256(folder/'checkpoint_final.pkl')!=summary['checkpoint_sha256'] or sha256(folder/'normalizer.json')!=summary['normalizer_sha256']:
            raise ValueError('trained checkpoint or normalizer changed')
    return selected,job,checked


def evaluation_ranges(users,batch):
    split=(users//2//batch)*batch
    return [(0,split),(split,users)] if split else [(0,users)]


def _baseline_digests(job,start,stop):
    rows={name:[] for name in ('Never','Always','Random')}
    seeds=[]
    for user in range(start,stop):
        seeds.append([user,episode_seed('evaluation',job['domain'],job['evaluation_seed'],user)])
        rng=np.random.default_rng(episode_seed('random-coupon-policy',job['domain'],job['evaluation_seed'],user))
        for name,actions in (('Never',[0]*job['test_weeks']),('Always',[1]*job['test_weeks']),
                             ('Random',rng.integers(2,size=job['test_weeks']).tolist())):
            rows[name].append([user,actions])
    return {name:protocol_digest(values) for name,values in rows.items()},protocol_digest(seeds)


def merge_shared_results(parts,job):
    """Union disjoint customer ranges; no trajectory is joined after divergence."""
    parts=sorted(parts,key=lambda part:part[1]['user_range'][0])
    if [tuple(protocol['user_range']) for _,protocol in parts]!=evaluation_ranges(job['n_test'],job['eval_batch_size']):
        raise ValueError('evaluation shards overlap, omit customers or change inference batch boundaries')
    common=('global_users','weeks','domain','evaluation_seed','policy_frozen','policy_fingerprints',
        'actor_inference_batch_size','sharing','score_recalculation','conversion_unit','random_baseline_seed',
        'initialization_sampler','policy_count','fixed_baselines','score_parameters','member_score_parameters')
    reference=parts[0][1]
    keys=set(parts[0][0]);attempts=[]
    if keys!={*reference['policy_fingerprints'],'Never','Always','Random'} or reference['policy_count']!=len(keys)-3:
        raise ValueError('evaluation shard omitted a scheduled policy or baseline')
    counters=('environment_resets','actual_warmup_resets','initial_state_cache_hits','environment_advances',
        'episode_clones','independent_environment_advances','transitions')
    protocol={name:reference[name] for name in common}
    if (not reference['policy_frozen'] or reference['global_users']!=job['n_test'] or reference['domain']!=job['domain']
            or reference['weeks']!=job['test_weeks'] or reference['evaluation_seed']!=job['evaluation_seed']
            or reference['actor_inference_batch_size']!=job['eval_batch_size']
            or reference['score_parameters']!=score_parameters(job)):
        raise ValueError('evaluation shards require frozen policies over the original complete cohort')
    for frames,part in parts:
        if set(frames)!=keys or any(part[name]!=reference[name] for name in common):
            raise ValueError('evaluation shard policy/world protocol differs')
        start,stop=part['user_range'];expected=set(range(start,stop))
        if part['users']!=stop-start or part['transitions']!=(stop-start)*job['test_weeks']:
            raise ValueError('evaluation shard budget differs')
        for frame in frames.values():
            if len(frame)!=stop-start or not frame.user.is_unique or set(frame.user)!=expected:
                raise ValueError('evaluation shard omitted or duplicated a scheduled customer')
        baselines,seeds=_baseline_digests(job,start,stop)
        if part['baseline_action_sequence_sha256']!=baselines or part['user_episode_seed_sha256']!=seeds:
            raise ValueError('evaluation shard changed the preassigned customer or baseline seeds')
        attempts.extend(part['initialization_attempts_by_user'])
    if len({row[0] for row in attempts})!=len(attempts) or any(not 0<=row[0]<job['n_test'] for row in attempts):
        raise ValueError('evaluation shard initialization diagnostics duplicate or invent customers')
    protocol.update({name:sum(part[name] for _,part in parts) for name in counters})
    if (protocol['environment_resets']!=job['n_test']
            or protocol['actual_warmup_resets']+protocol['initial_state_cache_hits']!=job['n_test']
            or protocol['independent_environment_advances']!=len(keys)*job['n_test']*job['test_weeks']):
        raise ValueError('evaluation shard environment counts do not cover the complete cohort')
    measured=[row[1] for row in attempts]
    protocol.update(users=job['n_test'],user_range=[0,job['n_test']],worker_seconds=sum(part['seconds'] for _,part in parts),
        initialization_attempts_by_user=sorted(attempts),initialization_attempts={
            'measured_users':len(measured),'maximum':max(measured) if measured else None,
            'p99':float(np.quantile(measured,.99)) if measured else None,
            'above_native_limit':sum(value>reference['initialization_sampler']['native_candidates'] for value in measured)},
        evaluation_shards=[part['user_range'] for _,part in parts])
    protocol['baseline_action_sequence_sha256'],protocol['user_episode_seed_sha256']=_baseline_digests(job,0,job['n_test'])
    return {key:pd.concat([frames[key] for frames,_ in parts],ignore_index=True).sort_values('user').reset_index(drop=True)
            for key in keys},protocol


def evaluate_domain_shard(matrix_dir,domain,user_start,user_stop):
    torch.set_num_threads(1)
    root=Path(matrix_dir);selected,job,checked=_domain_training(root,domain)
    if (user_start,user_stop) not in evaluation_ranges(job['n_test'],job['eval_batch_size']):
        raise ValueError('evaluation shard differs from its preassigned range')
    policies={}
    for key,summary in selected.items():
        folder=root/key
        with (folder/'checkpoint_final.pkl').open('rb') as stream:policy=pickle.load(stream)
        policies[key]={'policy':policy,'normalizer':json.loads((folder/'normalizer.json').read_text()),
            'cost':summary['job']['cost'],**score_parameters(summary['job'])}
    directory=root/'shared_evaluation'/domain/f'users_{user_start}_{user_stop}';directory.mkdir(parents=True,exist_ok=False)
    started_at=datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1):
        template=make_env(job,job['test_weeks'],cost='S')
        frames,protocol=evaluate_shared_policies(policies,template,job,
            user_start=user_start,user_stop=user_stop,
            progress=lambda users,seconds,counts:write_json(directory/'progress.json',{'users':users,'seconds':seconds,
                'user_range':[user_start,user_stop],**counts}))
    if validate_freeze(job['frozen_manifest'],job['configs'])!=checked:raise ValueError('world changed during frozen evaluation')
    for key,frame in frames.items():frame.to_parquet(directory/f'{key}.parquet',index=False)
    write_json(directory/'protocol.json',{'status':'complete','frozen_world':checked,'protocol':protocol,
        'started_at':started_at,'finished_at':datetime.now(timezone.utc).isoformat(),
        'frames':{key:sha256(directory/f'{key}.parquet') for key in frames},
        'source_training_summaries':{key:sha256(root/key/'training_summary.json') for key in selected}})
    return {'domain':domain,'user_range':[user_start,user_stop],'directory':str(directory.resolve()),
        'protocol_sha256':sha256(directory/'protocol.json')}


def evaluate_domain(matrix_dir,domain,shards):
    root=Path(matrix_dir);selected,job,checked=_domain_training(root,domain)
    parts=[];stamps=[]
    expected_keys={*selected,'Never','Always','Random'}
    training_hashes={key:sha256(root/key/'training_summary.json') for key in selected}
    for shard in shards:
        directory=Path(shard['directory']);path=directory/'protocol.json'
        if shard['domain']!=domain or sha256(path)!=shard['protocol_sha256']:raise ValueError('evaluation shard changed')
        block=json.loads(path.read_text())
        if (block['status']!='complete' or block['frozen_world']!=checked or set(block['frames'])!=expected_keys
                or block['source_training_summaries']!=training_hashes or block['protocol']['user_range']!=shard['user_range']):
            raise ValueError('evaluation shard checkpoint/world binding differs')
        frames={}
        for key,checksum in block['frames'].items():
            if sha256(directory/f'{key}.parquet')!=checksum:raise ValueError('evaluation shard rows changed')
            frames[key]=pd.read_parquet(directory/f'{key}.parquet')
        parts.append((frames,block['protocol']));stamps.append((block['started_at'],block['finished_at']))
    frames,protocol=merge_shared_results(parts,job)
    protocol['seconds']=(max(datetime.fromisoformat(end) for _,end in stamps)-min(datetime.fromisoformat(start) for start,_ in stamps)).total_seconds()
    protocol['timing']='domain wall time across shards; worker_seconds sums actual rollout times'
    protocol['shard_protocols']=shards
    directory=root/'shared_evaluation'/domain
    completed={}
    for key,summary in selected.items():
        frame=frames[key]
        if len(frame)!=job['n_test'] or not frame.user.is_unique or set(frame.user)!=set(range(job['n_test'])):
            raise ValueError('evaluation omitted or duplicated a scheduled customer')
        folder=root/key;frame.to_parquet(folder/'evaluation_users.parquet',index=False);frame.to_csv(folder/'evaluation_users.csv',index=False)
        summary.update(stage='complete',evaluation=summarize(frame),evaluation_protocol=protocol)
        write_json(folder/'summary.json',summary);completed[key]={'evaluation':summary['evaluation'],'learning':summary['learning']}
    baselines={}
    scoring=score_parameters(job)
    for key in ('Never','Always','Random'):
        frame=frames[key]
        frame.to_parquet(directory/f'{key.lower()}_users.parquet',index=False)
        baselines[key]={}
        for cost,field in (('S','sends'),('D','redemptions')):
            scored=frame.assign(reward_score=scoring['conversion_value']*frame.converted_sessions-scoring['coupon_cost']*frame[field])
            baselines[key][cost]=summarize(scored)
    write_json(directory/'summary.json',{'status':'complete','domain':domain,'evaluation_protocol':protocol,
        'fixed_policy_baselines':baselines,'frozen_world':checked,'source_training_summaries':{k:sha256(root/k/'training_summary.json') for k in selected}})
    return completed


def build_jobs(options,configs,output,freeze=None):
    domains=options['domains'];costs=options['costs'];agents=options['agents']
    scoring=score_parameters(options)
    jobs={}
    for cost in costs:
        for agent in agents:
            for domain in domains:
                key=f'{domain}_{cost}_{agent}_seed{options["seed"]}'
                jobs[key]={**{k:v for k,v in options.items() if k not in ('domains','costs','agents','stage','plan')},
                    **scoring,'domain':domain,'cost':cost,'agent':agent,'key':key,'config':configs[domain],
                    'configs':configs,'freeze':freeze,'output':str((Path(output)/key).resolve())}
    return jobs


def validate_evaluation_options(options,training):
    paired=('seed','n_train','n_test','train_weeks','test_weeks','calibration_users','calibration_seed','evaluation_seed','eval_batch_size')
    if (any(options[name]!=training[name] for name in paired)
            or score_parameters(options)!=score_parameters(training)):
        raise ValueError('evaluation budgets/seeds/batch boundaries/score weights must match the preassigned training matrix')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,required=True,help='JSON mapping cosmetics/electronics/tianchi to native canonical-enabled config paths')
    parser.add_argument('--frozen-manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true',help='Write budgets only; do not construct source worlds or learn')
    parser.add_argument('--stage',choices=('all','train','evaluate'),default='all')
    parser.add_argument('--domains',nargs='+',choices=DOMAINS,default=list(DOMAINS))
    parser.add_argument('--costs',nargs='+',choices=('S','D'),default=['S','D'])
    parser.add_argument('--agents',nargs='+',choices=list(AGENTS),default=list(AGENTS))
    parser.add_argument('--conversion-value',type=float,default=1000.,help='Positive simulated value per converted session/purchase day')
    parser.add_argument('--coupon-cost',type=float,default=100.,help='Nonnegative simulated cost per send (S) or redemption (D); coupon response stays fixed')
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='mps')
    for name,default in (('seed',42),('n-train',1000),('n-test',5000),('train-weeks',13),('test-weeks',52),
        ('calibration-users',128),('calibration-seed',7131),('evaluation-seed',95001),('learning-starts',1024),
        ('sac-batch-size',256),('ppo-rollout',256),('eval-batch-size',64),('progress-users',25),('workers',2)):
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--initial-state-cache',type=Path)
    args=parser.parse_args()
    try:score_parameters(vars(args))
    except ValueError as error:parser.error(str(error))
    assert_active_checkout()
    sizes=(args.n_train,args.n_test,args.train_weeks,args.test_weeks,args.calibration_users,args.learning_starts,
        args.sac_batch_size,args.ppo_rollout,args.eval_batch_size,args.progress_users,args.workers)
    if min(sizes)<1 or args.workers>2:parser.error('positive budgets and at most2 heavy workers are required')
    if min(args.seed,args.calibration_seed,args.evaluation_seed)<0:parser.error('nonnegative seeds required')
    if any(len(set(v))!=len(v) for v in (args.domains,args.costs,args.agents)):parser.error('matrix axes must be unique')
    if {'SAC','TD3'}&set(args.agents) and args.n_train*args.train_weeks<args.learning_starts:parser.error('off-policy budget does not reach replay warmup')
    supplied=json.loads(args.configs.read_text());configs={d:str(_path(supplied[d],args.configs.resolve().parent)) for d in args.domains}
    options={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('configs','output')}
    options['initial_state_cache']=str((args.initial_state_cache or args.output/'initial_state_cache').resolve())
    if args.plan:
        args.output.mkdir(parents=True,exist_ok=False)
        jobs=build_jobs(options,configs,args.output)
        plan={'status':'plan_only_no_training','jobs':jobs,'jobs_requested':len(jobs),'max_heavy_jobs':args.workers,
            'training_transitions_per_job':args.n_train*args.train_weeks,'evaluation_transitions_per_job':args.n_test*args.test_weeks,
            'normalizer_transitions_per_domain':args.calibration_users*args.train_weeks,'warmup_days':364,
            'evaluation_user_ranges':evaluation_ranges(args.n_test,args.eval_batch_size),
            'training_transitions_total':len(jobs)*args.n_train*args.train_weeks,'logical_evaluation_transitions_total':len(jobs)*args.n_test*args.test_weeks,
            'actual_learning_requires':'completed source/coupon/104w gates and exact frozen input SHA; no exception for a pilot'}
        write_json(args.output/'matrix_plan.json',plan);print(json.dumps({'status':plan['status'],'cells':len(jobs)}));return
    freeze=validate_freeze(args.frozen_manifest,configs)
    if args.stage!='evaluate':
        args.output.mkdir(parents=True,exist_ok=False)
        jobs=build_jobs(options,configs,args.output,freeze)
        matrix={'status':'prepared','jobs':jobs,'jobs_requested':len(jobs),'jobs_trained':{},'jobs_complete':{},'jobs_failed':{},
            'budget_class':'paper_scale' if (args.n_train,args.train_weeks,args.n_test,args.test_weeks,args.seed)==(1000,13,5000,52,42) else 'pilot',
            'protocol':options,'frozen_world':freeze,'runner_sha256':sha256(__file__)}
        write_json(args.output/'matrix_summary.json',matrix)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(train_job,job):key for key,job in jobs.items()}
            for future in as_completed(futures):
                key=futures[future]
                try:
                    result=future.result();matrix['jobs_trained'][key]={'training':result['training'],'learning':result['learning']}
                except Exception as error:matrix['jobs_failed'][key]=repr(error)
                write_json(args.output/'matrix_summary.json',matrix)
        if matrix['jobs_failed']:raise RuntimeError(f'policy cells failed without dropping customers: {matrix["jobs_failed"]}')
    else:
        matrix=json.loads((args.output/'matrix_summary.json').read_text())
        if matrix['frozen_world']!=freeze:raise ValueError('evaluation freeze differs from training')
        validate_evaluation_options(options,matrix['protocol'])
    if args.stage!='train':
        ranges=evaluation_ranges(args.n_test,args.eval_batch_size)
        matrix['evaluation_user_ranges']=ranges;matrix['evaluation_shards']={}
        write_json(args.output/'matrix_summary.json',matrix)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(evaluate_domain_shard,str(args.output.resolve()),domain,start,stop):(domain,start,stop)
                for domain in args.domains for start,stop in ranges}
            for future in as_completed(futures):
                domain,start,stop=futures[future];key=f'{domain}:{start}:{stop}'
                try:matrix['evaluation_shards'][key]=future.result()
                except Exception as error:matrix['jobs_failed']['evaluation:'+key]=repr(error)
                write_json(args.output/'matrix_summary.json',matrix)
        for domain in args.domains:
            shards=[value for value in matrix['evaluation_shards'].values() if value['domain']==domain]
            if len(shards)!=len(ranges):continue
            try:matrix['jobs_complete'].update(evaluate_domain(str(args.output.resolve()),domain,shards))
            except Exception as error:matrix['jobs_failed']['evaluation:'+domain]=repr(error)
            write_json(args.output/'matrix_summary.json',matrix)
        if matrix['jobs_failed']:raise RuntimeError(f'frozen evaluation failed: {matrix["jobs_failed"]}')
    matrix['status']='complete' if len(matrix['jobs_complete'])==len(matrix['jobs']) else 'trained'
    write_json(args.output/'matrix_summary.json',matrix)


if __name__=='__main__':main()
