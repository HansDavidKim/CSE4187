"""Main-world ports of the tested v6 weekly training/cache/prefix utilities.

Only trusted local cache files written by this runner may be loaded. Fitted
models and callbacks stay in the constructor; one pickled mutable graph keeps
RNG/active-row aliases intact. Native bounded reset is never extended/reseeded.
"""
from contextlib import contextmanager
from copy import copy,deepcopy
import csv
import fcntl
from functools import lru_cache
import gc
import hashlib
import inspect
import json
import marshal
from numbers import Real
import os
from pathlib import Path
import pickle
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.time_spec import weekly_metadata

TOTALS=("converted_sessions","orders","sends","redemptions","paid_minor","discount_minor","send_cost_minor")
UNOBSERVED_T=("orders","paid_minor","discount_minor")
_MARKER="_benchmark_initial_static_keys"


def score_parameters(values):
    result={}
    for name,default in (('conversion_value',1000.),('coupon_cost',100.)):
        value=values.get(name,default)
        if (isinstance(value,(bool,np.bool_)) or not isinstance(value,Real)
                or not np.isfinite(value) or value<0 or name=='conversion_value' and value==0):
            raise ValueError(f'{name} must be a finite {"positive" if name=="conversion_value" else "nonnegative"} number')
        result[name]=float(value)
    return result


@lru_cache(maxsize=4096)
def _file_digest(path,size,modified_ns):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def sha256(path):
    path=Path(path).resolve();stat=path.stat()
    return _file_digest(str(path),stat.st_size,stat.st_mtime_ns)


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def protocol_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fit_normalizer(frame):
    values=frame[list(EXPANDED_FEATURES)].to_numpy(dtype=float)
    finite=np.isfinite(values)
    median=np.array([np.median(values[finite[:,i],i]) if finite[:,i].any() else 0. for i in range(len(EXPANDED_FEATURES))])
    filled=np.where(finite,values,median)
    mean,scale=filled.mean(axis=0),filled.std(axis=0)
    scale[scale==0]=1.
    return {'features':list(EXPANDED_FEATURES),'median':median.tolist(),'mean':mean.tolist(),
            'scale':scale.tolist(),'clip':10.,'scope':'synthetic training calibration only; existing39; unsupported Tianchi observations remain unobserved in source records'}


def new_totals(user,domain):
    return {'user':user,'reward_score':0.,'conversion_unit':'purchase_day' if domain=='tianchi' else 'retail_session',**{key:None if domain=='tianchi' and key in UNOBSERVED_T else 0. for key in TOTALS}}


def add_totals(row,info,domain):
    for key in TOTALS:
        if key not in info:raise ValueError(f'weekly outcome omits {key}')
        value=info[key]
        if domain=='tianchi' and key in UNOBSERVED_T:
            if value is not None:raise ValueError(f'Tianchi {key} is unobserved; a numeric value invents retail evidence')
            row[key]=None
        else:
            if value is None or isinstance(value,(bool,np.bool_)) or not np.isfinite(float(value)) or value<0:
                raise ValueError(f'weekly observed total must be finite and nonnegative: {key}')
            row[key]+=float(value)


def summarize(frame):
    if frame.empty:raise ValueError('policy summaries require every scheduled customer')
    rewards=frame.reward_score.to_numpy(float)
    if not np.isfinite(rewards).all():raise ValueError('nonfinite policy reward')
    deviation=float(rewards.std(ddof=1)) if len(rewards)>1 else 0.
    error=deviation/np.sqrt(len(rewards))
    result={'users':len(frame),'mean_return':float(rewards.mean()),'std_return':deviation,
            'user_mean_standard_error':error,'user_mean_ci95':[float(rewards.mean()-1.96*error),float(rewards.mean()+1.96*error)]}
    for key in TOTALS:
        if frame[key].isna().any() and not frame[key].isna().all():raise ValueError(f'mixed outcome observability: {key}')
        result[f'mean_{key}']=None if frame[key].isna().all() else float(frame[key].mean())
    result['uncertainty']='conditional Monte Carlo user variation in one frozen synthetic world; not source-model, policy-training-seed or real-world causal uncertainty'
    return result


def _binding(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return {'dtype':str(value.dtype),'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,(list,tuple)):return [_binding(v) for v in value]
    if isinstance(value,dict):return {str(k):_binding(v) for k,v in value.items()}
    if isinstance(value,Path):return str(value.resolve())
    raise ValueError('cache callbacks must keep mutable customer state/RNG in the environment')


def _callback(function):
    if function is None:return None
    result={'module':function.__module__,'name':function.__qualname__,
            'code':hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
            'closure':[_callback(c.cell_contents) if callable(c.cell_contents) else _binding(c.cell_contents)
                       for c in (function.__closure__ or ())]}
    owner=getattr(function,'__self__',None)
    if owner is not None:
        if not hasattr(owner,'_model'):raise ValueError('bound cache callback needs an immutable model binding')
        result['model']=_binding(owner._model)
    return result


def sampler_provenance(env=None):
    return {'type':'native_bounded_past_only_reset','continuation_or_seed_replacement':False,'native_candidates':getattr(env,'INITIAL_CANDIDATES',1000),
            'eligibility':'native generated history age and past28 activity only; no future survival filter',
            'warmup_days':getattr(env,'warmup_days',None),'helper_sha256':sha256(__file__)}


def _key(env,seed):
    guard=protocol_digest({'config':env.config,'models':env.model_shas,'horizon':env.horizon_days,
        'warmup':env.warmup_days,'history':env.min_history_days,'coupon':env.coupon_valid_days})
    if hasattr(env,_MARKER) and guard!=env._benchmark_cache_guard:
        raise ValueError('constructor inputs changed after initial-cache binding')
    if not hasattr(env,_MARKER):
        if hasattr(env,'records'):raise ValueError('initial-state caching must start before the first reset')
        cls=type(env)
        signature={'version':1,'class':cls.__module__+':'+cls.__qualname__,
            'environment_sha256':sha256(inspect.getfile(cls)),'helper_sha256':sha256(__file__),
            'freeze':getattr(env,'_benchmark_freeze_sha256',None),'config':env.config,'models':env.model_shas,
            'horizon_days':env.horizon_days,'warmup_days':env.warmup_days,'min_history_days':env.min_history_days,
            'coupon_valid_days':env.coupon_valid_days,
            'profile_ids':[p['profile_id'] for p in getattr(env,'profiles',())],
            'profile_weights':hashlib.sha256(np.asarray(getattr(env,'profile_probabilities',())).tobytes()).hexdigest(),
            'callbacks':{name:_callback(getattr(env,name,None)) for name in ('feature_builder','response_sampler','runtime_reward_offsets')}}
        env._benchmark_cache_signature=protocol_digest(signature)
        env._benchmark_cache_guard=guard
        setattr(env,_MARKER,tuple(env.__dict__))
    return protocol_digest({'base':env._benchmark_cache_signature,'seed':int(seed)})


def cache_reset(env,seed,cache_dir=None):
    if cache_dir is None:
        state=env.reset(seed);env._benchmark_last_cache_hit=False
        return state
    # Aggregate reset regenerates home_groups; it is not a constructor asset.
    key=_key(env,seed);static=(set(getattr(env,_MARKER))|{_MARKER})-{'home_groups'}
    directory=Path(cache_dir);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    destination=directory/f'{key}.pkl';checksum=directory/f'{key}.json'
    with (directory/f'{key}.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if destination.exists():
            data=destination.read_bytes()
            metadata=json.loads(checksum.read_text())
            if metadata!={'key':key,'sha256':hashlib.sha256(data).hexdigest()}:raise ValueError('private reset cache checksum changed')
            payload=pickle.loads(data)
            if payload['key']!=key or set(payload['dynamic'])&static or payload['sampler']!=sampler_provenance(env):
                raise ValueError('private reset cache signature/state changed')
            for name in set(env.__dict__)-static:delattr(env,name)
            env.__dict__.update(payload['dynamic'])
            if 'products' in env.records:env.records['products']=list(env.product_rows)
            if hasattr(env,'_basket_banks'):env._basket_banks={}
            env.records['_metadata']['cost_model']=env.cost_model
            for name in ('score_conversion_value','score_coupon_cost'):
                if hasattr(env,name):env.records['_metadata'][name]=getattr(env,name)
            actual=np.asarray(env._features(),float)
            if actual.shape!=(39,) or not np.array_equal(actual,payload['features'],equal_nan=True):
                raise ValueError('cached reset does not reproduce raw39')
            env._benchmark_last_cache_hit=True
            return actual
        state=np.asarray(env.reset(seed),float)
        env._benchmark_last_cache_hit=False
        if state.shape!=(39,):raise ValueError('cached reset requires raw39')
        dynamic={name:value for name,value in env.__dict__.items() if name not in static}
        if 'products' in dynamic['records']:dynamic['records']={**dynamic['records'],'products':None}
        payload={'key':key,'dynamic':dynamic,'features':state.copy(),'sampler':sampler_provenance(env)}
        data=pickle.dumps(payload,protocol=pickle.HIGHEST_PROTOCOL)
        descriptor,temporary=tempfile.mkstemp(prefix=f'.{key}.',suffix='.tmp',dir=directory)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
            os.link(temporary,destination)
            write_json(checksum,{'key':key,'sha256':hashlib.sha256(data).hexdigest()})
        finally:Path(temporary).unlink(missing_ok=True)
        return state


def clone_episode(env):
    # Native copy implementations preserve all mutable aliases. Only deterministic
    # fitted-template caches may be shared in addition to their immutable assets.
    memo={id(env._basket_banks):env._basket_banks} if hasattr(env,'_basket_banks') else {}
    if hasattr(env,'home_groups'):memo[id(env.home_groups)]=env.home_groups
    products=getattr(env,'records',{}).get('products')
    if products is not None:memo[id(products)]=list(products)
    return deepcopy(env,memo)


def policy_fingerprint(policy):
    def pack(value):
        if isinstance(value,torch.nn.Module):return pack(value.state_dict())
        if isinstance(value,torch.optim.Optimizer):return pack(value.state_dict())
        if isinstance(value,torch.Tensor):return pack(value.detach().cpu().numpy())
        if isinstance(value,np.ndarray):
            if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('policy parameters, optimizer state or stored inputs are nonfinite')
            return _binding(value)
        if isinstance(value,torch.device):return str(value)
        if isinstance(value,np.random.Generator):return pack(value.bit_generator.state)
        if isinstance(value,dict):return {str(k):pack(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)) or type(value).__name__=='deque':return [pack(v) for v in value]
        return _binding(value)
    return protocol_digest(pack(policy.__dict__))


def episode_seed(phase, domain, seed, user):
    value = json.dumps(["weekly-coupon-v1", phase, domain, int(seed), int(user)]).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def scale_states(states, normalizer):
    values = np.asarray(states, dtype=float)
    if values.shape[-1] != len(EXPANDED_FEATURES):
        raise ValueError("weekly policy observations must have exactly 39 features")
    filled = np.where(np.isfinite(values), values, normalizer["median"])
    with np.errstate(over="ignore"):
        scaled = np.clip((filled - normalizer["mean"]) / normalizer["scale"],
                         -normalizer["clip"], normalizer["clip"]).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("weekly normalized states must be finite")
    return scaled


def frozen_actions(policy, states):
    """Batch the exact greedy scores used by these agents' training=False path."""
    if hasattr(policy, "actor_inference"):
        with torch.no_grad():
            output = policy.actor_inference(torch.as_tensor(states, dtype=torch.float32))
            if not torch.isfinite(output).all():raise ValueError("nonfinite frozen policy scores")
            return ((output[:, 0] >= .5).long() if hasattr(policy, 'actor_updates') else output.argmax(-1)).numpy()
    if hasattr(policy, "a_inverse"):
        internal = np.column_stack((states, np.ones(len(states)))) if policy.b.shape[1] == states.shape[1] + 1 else states
        means = np.stack([policy.a_inverse[a] @ policy.b[a] for a in (0, 1)])
        return (internal @ means.T).argmax(axis=1)
    if hasattr(policy, "precision"):
        return np.full(len(states), int(np.argmax(policy.total / policy.precision)))
    if hasattr(policy, "count"):
        return np.full(len(states), int(np.argmax(policy.total / np.maximum(policy.count, 1))))
    return np.array([policy.act(state, training=False) for state in states], dtype=int)


@contextmanager
def episode_gc():
    """Collect in bounded episode batches instead of scanning every history."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        gc.collect()
        if enabled:
            gc.enable()


@episode_gc()
def fit_calibration(template, users, weeks, domain, seed, cache_dir=None):
    if cache_dir is None:
        return _fit_calibration(template, users, weeks, domain, seed)
    signature = {'environment': _key(template, episode_seed('calibration', domain, seed, 0)),
                 'users': users, 'weeks': weeks, 'domain': domain, 'seed': seed,
                 'runner_sha256': sha256(__file__)}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f'normalizer-{key}.json'
    with (directory / f'normalizer-{key}.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
            saved = json.loads(destination.read_text())
            result = saved['normalizer']
            digest = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
            if saved['signature'] != signature or saved['normalizer_sha256'] != digest:
                raise ValueError('shared normalizer signature or checksum differs')
            return result
        result = _fit_calibration(template, users, weeks, domain, seed, cache_dir,
                                  directory / f'normalizer-{key}.progress.json')
        digest = hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest()
        write_json(destination, {'signature': signature, 'normalizer': result, 'normalizer_sha256': digest})
        return result


def _fit_calibration(template, users, weeks, domain, seed, cache_dir=None, progress=None):
    env, states = copy(template), []
    rng = np.random.default_rng(seed)
    for user in range(users):
        state = cache_reset(env,episode_seed("calibration", domain, seed, user),cache_dir)
        for week in range(weeks):
            states.append(np.asarray(state).copy())
            state, _, done, _ = env.advance(int(rng.integers(2)), 7)
            if bool(done) != (week == weeks - 1):
                raise ValueError("calibration episodes must retain the complete weekly horizon")
        if progress is not None:
            write_json(progress, {'completed_users': user + 1, 'total_users': users,
                                  'transitions': len(states), 'weeks_per_user': weeks})
        if (user+1)%64==0:gc.collect()
    result = fit_normalizer(pd.DataFrame(np.asarray(states), columns=EXPANDED_FEATURES))
    result.update(calibration_users=users, calibration_transitions=len(states), calibration_seed=seed,
                  policy_training_feedback=False)
    result.update(weekly_metadata())
    return result


def gradient_counts(policy, ppo_learn_calls):
    updates = int(getattr(policy, "gradient_steps", 0))
    delayed = int(getattr(policy, 'actor_updates', 0))
    td3 = hasattr(policy, 'actor_updates')
    return {"sac_gradient_updates": 0 if td3 else updates, 'td3_critic_updates': updates if td3 else 0,
            'td3_actor_updates': delayed, "ppo_learn_calls": ppo_learn_calls,
            "optimizer_steps": updates+delayed if td3 else 2*updates+4*ppo_learn_calls}


@episode_gc()
def train_policy(policy, name, template, normalizer, job, output):
    if score_parameters(job)!=score_parameters({'conversion_value':getattr(template,'score_conversion_value',1000.),
                                              'coupon_cost':getattr(template,'score_coupon_cost',100.)}):
        raise ValueError('training environment score weights differ from the scheduled job')
    env = copy(template)
    rows, steps, ppo_calls, agent_seconds = [], 0, 0, 0.
    gc_calls,gc_seconds=0,0.
    total_steps = job["n_train"] * job["train_weeks"]
    started = perf_counter()
    curves = output / "training_curve.csv"
    with curves.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["steps", "users", "mean_return", "recent_100_mean_return",
                                                    "mean_sends", "sac_gradient_updates", 'td3_critic_updates',
                                                    'td3_actor_updates', "ppo_learn_calls", "optimizer_steps", "seconds"])
        writer.writeheader()
        for user in range(job["n_train"]):
            state = cache_reset(env,episode_seed("training", job["domain"], job["seed"], user),job.get("initial_state_cache"))
            totals = new_totals(user,job["domain"])
            for week in range(job["train_weeks"]):
                normalized = scale_states(state, normalizer)
                before = perf_counter()
                action = int(policy.act(normalized, training=True))
                agent_seconds += perf_counter() - before
                if action not in (0, 1):
                    raise ValueError("coupon action must be binary")
                next_state, reward, done, info = env.advance(action, 7)
                if bool(done) != (week == job["train_weeks"] - 1):
                    raise ValueError("training episodes must retain all scheduled weekly decisions")
                if not np.isfinite(reward):
                    raise ValueError("weekly training reward must be finite")
                before = perf_counter()
                policy.update(normalized, action, reward, scale_states(next_state, normalizer), bool(done))
                steps += 1
                if name == "PPO" and steps % job["ppo_rollout"] == 0:
                    policy.learn()
                    ppo_calls += 1
                agent_seconds += perf_counter() - before
                totals["reward_score"] += float(reward)
                add_totals(totals,info,job["domain"])
                state = next_state
            rows.append(totals)
            if name == "PPO" and user + 1 == job["n_train"] and policy.buffer:
                before = perf_counter(); policy.learn(); ppo_calls += 1
                agent_seconds += perf_counter() - before
            if (user+1)%64==0:
                before=perf_counter();gc.collect();gc_calls+=1;gc_seconds+=perf_counter()-before
            if (user + 1) % job["progress_users"] == 0 or user + 1 == job["n_train"]:
                returns = [row["reward_score"] for row in rows]
                progress = {"steps": steps, "users": user + 1, "mean_return": float(np.mean(returns)),
                            "recent_100_mean_return": float(np.mean(returns[-100:])),
                            "mean_sends": float(np.mean([row["sends"] for row in rows])),
                            **gradient_counts(policy, ppo_calls), "seconds": perf_counter() - started}
                writer.writerow(progress)
                handle.flush()
                write_json(output / "progress.json", {"stage": "training", **progress, "total_steps": total_steps})
                print(json.dumps({"job": job["key"], "stage": "training", **progress}), flush=True)
    if hasattr(policy, "sync_policy"):
        policy.sync_policy()
    counts = gradient_counts(policy, ppo_calls)
    fingerprint=policy_fingerprint(policy)
    if name in {"PPO", "SAC", 'TD3'} and counts["optimizer_steps"] == 0:
        raise ValueError("neural policy completed no optimizer steps")
    with (output / "checkpoint_final.pkl").open("xb") as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "training_users.parquet", index=False)
    return frame, {"transitions": steps, "policy_fingerprint": fingerprint, **counts, "analytical_policy_updates": steps if name in {"UCB","TS","LinUCB","LinTS"} else 0, "agent_seconds": agent_seconds,
                   "total_seconds": perf_counter() - started,'periodic_gc_collect_calls':gc_calls,'periodic_gc_seconds':gc_seconds,
                   'gc_schedule':'every64 completed customers plus one final collection; final collection excluded from periodic counters',
                   "sac_learning_starts": job["learning_starts"] if name == "SAC" else None,
                   "sac_train_interval": 1 if name == "SAC" else None,
                   "sac_batch_size": job["sac_batch_size"] if name == "SAC" else None,
                   "ppo_rollout": job["ppo_rollout"] if name == "PPO" else None,
                   "ppo_epochs_per_rollout": 4 if name == "PPO" else None,
                   "off_policy_learning_starts": job["learning_starts"] if name in {"SAC","TD3"} else None,
                   "off_policy_batch_size": job["sac_batch_size"] if name in {"SAC","TD3"} else None,
                   "off_policy_train_interval": 1 if name in {"SAC","TD3"} else None,
                   "optimizer_device": str(getattr(policy,"device","analytical_numpy"))}


def evaluate_policy(policy, template, normalizer, job, output):
    started, rows = perf_counter(), []
    for first in range(0, job["n_test"], job["eval_batch_size"]):
        users = list(range(first, min(first + job["eval_batch_size"], job["n_test"])))
        envs = [copy(template) for _ in users]
        states = [cache_reset(env,episode_seed("evaluation", job["domain"], job["evaluation_seed"], user),job.get("initial_state_cache"))
                  for env, user in zip(envs, users)]
        totals = [new_totals(user,job["domain"]) for user in users]
        for week in range(job["test_weeks"]):
            actions = frozen_actions(policy, scale_states(states, normalizer))
            for i, (env, action) in enumerate(zip(envs, actions)):
                state, reward, done, info = env.advance(int(action), 7)
                if bool(done) != (week == job["test_weeks"] - 1) or not np.isfinite(reward):
                    raise ValueError("evaluation requires finite rewards over the complete weekly horizon")
                states[i] = state
                totals[i]["reward_score"] += float(reward)
                add_totals(totals[i],info,job["domain"])
        rows.extend(totals)
        write_json(output / "progress.json", {"stage": "frozen_evaluation", "users": len(rows),
                   "total_users": job["n_test"], "transitions": len(rows) * job["test_weeks"],
                   "seconds": perf_counter() - started})
        print(json.dumps({"job": job["key"], "stage": "frozen_evaluation", "users": len(rows),
                          "total_users": job["n_test"], "seconds": perf_counter() - started}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output / "evaluation_users.parquet", index=False)
    frame.to_csv(output / "evaluation_users.csv", index=False)
    return frame, {"transitions": len(rows) * job["test_weeks"], "seconds": perf_counter() - started,
                   "policy_frozen": True, "actor_inference_batch_size": job["eval_batch_size"]}




def evaluate_shared_policies(policies, template, job, progress=None, *, user_start=0, user_stop=None):
    """Each member has a frozen policy, normalizer and S/D score definition.

    Nodes split before advancing when actions diverge and are never merged later.
    Never/Always/Random use the same environment tree and evaluation seeds.
    """
    if set(policies) & {"Never", "Always", "Random"}:
        raise ValueError("fixed baseline names are reserved")
    before_policies={key:policy_fingerprint(member["policy"]) for key,member in policies.items()}
    sampler = sampler_provenance(template if hasattr(template, 'settings') else None)
    user_stop=job["n_test"] if user_stop is None else user_stop
    batch=job["eval_batch_size"]
    if (type(user_start) is not int or type(user_stop) is not int or not 0<=user_start<user_stop<=job["n_test"]
            or user_start%batch or user_stop!=job["n_test"] and user_stop%batch):
        raise ValueError("evaluation user ranges must preserve global inference-batch boundaries")
    count_users=user_stop-user_start
    members = {**policies, "Never":{"fixed":0,"cost":"S"},
               "Always":{"fixed":1,"cost":"S"}, "Random":{"fixed":"random","cost":"S"}}
    scoring=score_parameters(job)
    member_scoring={key:score_parameters({**scoring,**member}) for key,member in members.items()}
    if any(member['cost'] not in ('S','D') for member in members.values()):
        raise ValueError('shared policy score cost model must be S or D')
    rows = {key:[] for key in members}
    baseline_action_rows={key:[] for key in ("Never","Always","Random")}
    initialization_attempts=[]
    initialization_attempt_rows=[]
    counts = {"environment_resets":0,"actual_warmup_resets":0,"initial_state_cache_hits":0,"environment_advances":0,"episode_clones":0}
    started = perf_counter()
    for first in range(user_start,user_stop,batch):
        users = list(range(first,min(first+batch,user_stop)))
        nodes, totals, random_actions = [], [], []
        actual_actions=[{key:[] for key in baseline_action_rows} for _ in users]
        for user in users:
            env = copy(template)
            state = cache_reset(env,episode_seed("evaluation",job["domain"],job["evaluation_seed"],user),job.get("initial_state_cache"))
            metadata=env.records.get("_metadata",{}) if isinstance(getattr(env,"records",None),dict) else {}
            if metadata.get("initial_candidate_attempts") is not None:
                initialization_attempts.append(int(metadata["initial_candidate_attempts"]))
                initialization_attempt_rows.append([user,int(metadata["initial_candidate_attempts"])])
            counts["environment_resets"] += 1
            counts["initial_state_cache_hits" if env._benchmark_last_cache_hit else "actual_warmup_resets"] += 1
            nodes.append([{"env":env,"state":state,"members":tuple(members)}])
            totals.append({key:new_totals(user,job["domain"]) for key in members})
            rng = np.random.default_rng(episode_seed("random-coupon-policy",job["domain"],job["evaluation_seed"],user))
            random_actions.append(rng.integers(2,size=job["test_weeks"]))
        for week in range(job["test_weeks"]):
            locations = [{key:node for node in user_nodes for key in node["members"]} for user_nodes in nodes]
            actions = {}
            for key, member in members.items():
                if "fixed" in member:
                    actions[key] = ([int(values[week]) for values in random_actions] if member["fixed"]=="random"
                                    else [member["fixed"]]*len(users))
                else:
                    states = [location[key]["state"] for location in locations]
                    actions[key] = frozen_actions(member["policy"],scale_states(states,member["normalizer"]))
                if len(actions[key]) != len(users) or any(int(a) not in (0,1) for a in actions[key]):
                    raise ValueError("frozen policies must produce one binary action per user")
                if key in baseline_action_rows:
                    for i,action in enumerate(actions[key]):actual_actions[i][key].append(int(action))
            for i, user_nodes in enumerate(nodes):
                next_nodes=[]
                for node in user_nodes:
                    groups={action:tuple(key for key in node["members"] if int(actions[key][i])==action) for action in (0,1)}
                    groups={action:keys for action,keys in groups.items() if keys}
                    # Make every branch before mutating the original environment.
                    branches={action:node["env"] if number==0 else clone_episode(node["env"])
                              for number,action in enumerate(groups)}
                    counts["episode_clones"] += len(branches)-1
                    for action, keys in groups.items():
                        env=branches[action]
                        state,reward,done,info=env.advance(action,7)
                        if bool(done)!=(week==job["test_weeks"]-1) or not np.isfinite(reward):
                            raise ValueError("shared evaluation requires finite rewards and complete weekly horizons")
                        counts["environment_advances"] += 1
                        for key in keys:
                            row=totals[i][key]
                            cost=info["sends"] if members[key]["cost"]=="S" else info["redemptions"]
                            weights=member_scoring[key]
                            row["reward_score"] += weights['conversion_value']*info["converted_sessions"]-weights['coupon_cost']*cost
                            add_totals(row,info,job["domain"])
                        next_nodes.append({"env":env,"state":state,"members":keys})
                nodes[i]=next_nodes
        for user_totals in totals:
            for key,row in user_totals.items():rows[key].append(row)
        for i,user in enumerate(users):
            for key in baseline_action_rows:baseline_action_rows[key].append([user,actual_actions[i][key]])
        nodes.clear();locations.clear()
        if progress is not None:progress(len(rows["Never"]),perf_counter()-started,dict(counts))
    if before_policies!={key:policy_fingerprint(member["policy"]) for key,member in policies.items()}:
        raise ValueError("frozen evaluation changed a policy or optimizer/RNG state")
    return {key:pd.DataFrame(values) for key,values in rows.items()}, {
        "transitions":count_users*job["test_weeks"],"seconds":perf_counter()-started,
        "users":count_users,"global_users":job["n_test"],"user_range":[user_start,user_stop],
        "weeks":job["test_weeks"],"domain":job["domain"],"evaluation_seed":job["evaluation_seed"],
        "policy_frozen":True,"policy_fingerprints":before_policies,"actor_inference_batch_size":job["eval_batch_size"],
        "sharing":"exact identical action prefixes only; no merging after divergence",
        "score_parameters":scoring,"member_score_parameters":member_scoring,
        "score_recalculation":f"S={scoring['conversion_value']:g} converted_sessions-{scoring['coupon_cost']:g} sends; D={scoring['conversion_value']:g} converted_sessions-{scoring['coupon_cost']:g} redemptions",
        "conversion_unit":"purchase_day" if job["domain"]=="tianchi" else "retail_session",
        "random_baseline_seed":"episode_seed(random-coupon-policy,domain,evaluation_seed,user)",
        "initialization_sampler":sampler,
        "initialization_attempts_by_user":initialization_attempt_rows,
        "initialization_attempts":{"measured_users":len(initialization_attempts),
            "maximum":max(initialization_attempts) if initialization_attempts else None,
            "p99":float(np.quantile(initialization_attempts,.99)) if initialization_attempts else None,
            "above_native_limit":sum(value>sampler["native_candidates"] for value in initialization_attempts)},
        "baseline_action_sequence_sha256":{key:protocol_digest(values) for key,values in baseline_action_rows.items()},
        "user_episode_seed_sha256":protocol_digest([[user,episode_seed("evaluation",job["domain"],job["evaluation_seed"],user)] for user in range(user_start,user_stop)]),
        "policy_count":len(policies),"fixed_baselines":3,**counts,
        "independent_environment_advances":len(members)*count_users*job["test_weeks"],
    }
