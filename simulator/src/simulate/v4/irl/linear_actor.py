"""Compile persona-specific KIT39+internal40 linear rewards onto Main's MDP."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from simulate.v4.features import EXPANDED_FEATURES
from . import fit_latent_shopping as shopping
from . import fit_joint_latent_shopping as joint
from .feature_reward import fit_feature_reward
from .source_contract import file_sha, source_profile_id

FEATURES = ('intercept', *(f'public:{x}' for x in EXPANDED_FEATURES),
            *(f'internal:{x}' for x in shopping.FEATURE_NAMES))


def record_indices(data, session_mask):
    selected = data['sessions'].loc[session_mask]
    if selected.empty:
        raise ValueError('no complete supported source sessions in the fit partition')
    return np.concatenate([np.arange(r.record_start,r.record_end) for r in selected.itertuples()])


def load_observations(directory, data, prepared_source):
    """Reuse only byte-bound retail measurements with exactly aligned decisions."""
    folder=Path(directory); descriptor=json.loads((folder/'manifest.json').read_text())
    if descriptor['domain'] not in ('cosmetics','electronics'):
        raise ValueError('daily Tianchi must not reuse legacy session decision caches')
    if descriptor['source_cache_sha256'] != file_sha(prepared_source):
        raise ValueError('observation cache refers to a different prepared source')
    if descriptor['feature_names'] != list(EXPANDED_FEATURES) or descriptor['core_feature_names'] != list(shopping.FEATURE_NAMES):
        raise ValueError('registered39 or internal40 order changed')
    if descriptor['feature_windows_days'] != [28,56,91]:
        raise ValueError('weekly source fitting requires4/8/13-week measurement windows')
    for name, expected in descriptor['files'].items():
        if file_sha(folder/name) != expected:
            raise ValueError('observation cache file differs from its saved SHA256')
    recorded=pd.read_parquet(folder/'records.parquet')
    keys=['user_id','session','at','target_at','action','step']
    left=data['records'][keys].copy();right=recorded[keys].copy()
    for frame in (left,right): frame['user_id']=frame.user_id.astype(str)
    pd.testing.assert_frame_equal(left.reset_index(drop=True),right.reset_index(drop=True),check_dtype=False,check_exact=True)
    with np.load(folder/'decision_features.npz',allow_pickle=False) as source:
        raw=source['observations'].copy()
        core=source['core_features'].copy()
        usable=source['dynamics_usable'].copy()
    if raw.shape != (len(left),39) or np.isinf(raw).any():
        raise ValueError('registered observation cache must be aligned finite-or-missing39')
    available=data.get('available_actions',shopping.ACTIONS[:-1])
    # Old caches encoded zero removes as observed; the new source contract masks them.
    for action, index in [('cart',0),('remove',1)]:
        if action not in available:
            core[:,[shopping.FEATURE_NAMES.index(f'history_{index}_{part}') for part in ('value','observed')]]=0.
    np.testing.assert_allclose(data['features'],core,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(data['dynamics_usable'],usable)
    return raw, {'directory':str(folder.resolve()),'manifest_sha256':file_sha(folder/'manifest.json'),
                 'source_sha256':file_sha(prepared_source),'records_exact':True,'core40_exact_after_declared_support':True,
                 'public39_measurements_reused_without_recalculation':True}


def fit_transform(core, raw):
    if core.ndim!=2 or core.shape[1]!=40 or raw.shape!=(len(core),39) or not np.isfinite(core).all() or np.isinf(raw).any():
        raise ValueError('aligned existing40 and finite-or-missing39 are required')
    logged=np.sign(raw)*np.log1p(np.abs(raw)); observed=np.isfinite(logged)
    support=observed.any(axis=0)
    median=np.array([np.median(logged[observed[:,i],i]) if support[i] else 0. for i in range(39)])
    filled=np.where(observed,logged,median)
    mean,scale=filled.mean(axis=0),filled.std(axis=0);scale[scale<1e-12]=1.
    cmean,cscale=core.mean(axis=0),core.std(axis=0);cscale[cscale<1e-12]=1.
    return {'public_names':list(EXPANDED_FEATURES),'internal_names':list(shopping.FEATURE_NAMES),
        'public_median':median.tolist(),'public_mean':mean.tolist(),'public_scale':scale.tolist(),
        'public_supported':support.tolist(),'core_mean':cmean.tolist(),'core_scale':cscale.tolist(),
        'training_rows':len(core),'missing_value_rule':'training-median imputation; unsupported coefficients have zero basis; no missing-indicator predictors'}


def transformed(core, raw, transform):
    core=np.asarray(core,float);raw=np.asarray(raw,float)
    if core.shape!=(len(raw),40) or raw.ndim!=2 or raw.shape[1]!=39 or not np.isfinite(core).all() or np.isinf(raw).any():
        raise ValueError('invalid existing39+40 input shape or values')
    logged=np.sign(raw)*np.log1p(np.abs(raw))
    values=(np.where(np.isfinite(raw),logged,transform['public_median'])-transform['public_mean'])/transform['public_scale']
    values[:,~np.asarray(transform['public_supported'],bool)]=0.
    return np.column_stack([np.ones(len(raw)),values,(core-transform['core_mean'])/transform['core_scale']])


def encode_records(encoder, features, records):
    if encoder.get('lifecycle_memory'):
        from .source_phases import phase_ids
        return shopping.encode(encoder,features,phase_ids=phase_ids(records))
    return shopping.encode(encoder,features)


def state_basis(data, raw, training_mask, encoder):
    indices=record_indices(data,training_mask)
    transform=fit_transform(data['features'][indices],raw[indices])
    states,fallback=encode_records(encoder,data['features'][indices],data['records'].iloc[indices])
    if fallback.any(): raise ValueError('training state partition cannot need fallback')
    x=transformed(data['features'][indices],raw[indices],transform)
    count=np.bincount(states,minlength=len(encoder['state_keys']))
    if (count==0).any(): raise ValueError('every reward state must be observed in training')
    basis=np.column_stack([np.bincount(states,weights=x[:,j],minlength=len(count))/count for j in range(80)])
    phases=(np.asarray(encoder['state_keys'])[:,2] if encoder.get('lifecycle_memory') else np.zeros(len(basis),int))
    blocks=1+int(phases.max())
    expanded=np.column_stack([basis*(phases==phase)[:,None] for phase in range(blocks)])
    names=[f'phase{phase}:{name}' for phase in range(blocks) for name in FEATURES]
    return basis,expanded,names,phases,transform


def _return_offset(activity, regularization):
    if activity.empty: return 0., {'observations':0,'estimated':False,'fallback':'pooled source return GLM'}
    x=activity.base_logit.to_numpy(float);y=activity.label.to_numpy(float)
    def objective(value):
        eta=x+value[0];p=1/(1+np.exp(-np.clip(eta,-700,700)))
        return float(np.mean(np.logaddexp(0,eta)-y*eta)+regularization*value[0]**2/2), np.array([np.mean(p-y)+regularization*value[0]])
    result=minimize(objective,[0.],jac=True,method='L-BFGS-B',options={'gtol':1e-9,'ftol':1e-12})
    if not result.success: raise ValueError(f'persona return offset did not converge: {result.message}')
    return float(result.x[0]), {'observations':len(activity),'estimated':True,'objective':float(result.fun),'converged':True}


def fit_grouped_actor(data, raw, assignments, base_model, return_model, return_sha, *,
                      regularization=.01, selected_penalties=None,max_iter=1000,prior_mode=None):
    if prior_mode not in (None,'pooled_feature_reward'):
        raise ValueError('prior_mode must be None or pooled_feature_reward')
    contract=base_model['source_contract'];cutoff=pd.Timestamp(contract['training_cutoff_at'])
    if pd.Timestamp(return_model['validation']['temporal_cutoff'])!=cutoff:
        raise ValueError('linear actor and return model must have the same fit cutoff')
    groups=assignments.copy();groups['user_id']=groups.user_id.astype(str)
    if groups.user_id.duplicated().any() or not groups.assignment_cutoff.eq(cutoff).all():
        raise ValueError('unique frozen persona assignments at the fit cutoff are required')
    labels=groups.set_index('user_id').cluster.astype(str)
    outer,_,split=shopping.old.split_sessions(data,cutoff,pd.Timestamp(contract['inner_cutoff_at']),'known-total')
    train=outer['train'].copy()
    inner_fit=cutoff==pd.Timestamp(contract['inner_cutoff_at'])
    if inner_fit: train &= data['sessions'].inner_user
    idx=record_indices(data,train);panel=shopping.select_panel(data,train)
    memory=data.get('lifecycle_memory')
    encoder=shopping.fit_encoder(panel,available_actions=contract['observed_actions'],lifecycle_memory=memory)
    matrices,mask,state,dynamics=shopping.fit_dynamics(encoder,panel)
    basis,expanded,names,phases,transform=state_basis(data,raw,train,encoder)
    panel_labels=panel['records'].user_id.astype(str).map(labels)
    if panel_labels.isna().any(): raise ValueError('training action user lacks a frozen persona')
    activity,activity_audit=joint.build_activity_panel(data['source_rows'],return_model,int(base_model['seed']))
    eligible=activity.train_user & activity.target_at.lt(cutoff)
    if inner_fit: eligible &= activity.user_id.map(lambda uid:hashlib.sha256(f"{base_model['seed']}:inner:{uid}".encode()).digest()[0]<205)
    activity=activity.loc[eligible].copy();activity['persona']=activity.user_id.astype(str).map(labels)
    if activity.persona.isna().any(): raise ValueError('training return user lacks a frozen persona')
    prefix_users=groups.loc[groups.user_id.map(lambda uid:hashlib.sha256(f"{base_model['seed']}:{uid}".encode()).digest()[0]<205)]
    if inner_fit: prefix_users=prefix_users.loc[prefix_users.user_id.map(lambda uid:hashlib.sha256(f"{base_model['seed']}:inner:{uid}".encode()).digest()[0]<205)]
    prior=prefix_users.cluster.astype(str).value_counts().sort_index()
    personas=list(prior.index);weights=(prior/prior.sum()).to_numpy()
    all_counts=np.bincount(state*5+panel['records'].action.to_numpy(int),minlength=len(mask)*5).reshape(len(mask),5)
    pooled={};rewards=[];policies=[];values=[];offsets=[];coefficients=[];reports={}
    for label in personas:
        keep=panel_labels.eq(label).to_numpy()
        counts=np.bincount(state[keep]*5+panel['records'].loc[keep,'action'].to_numpy(int),minlength=len(mask)*5).reshape(len(mask),5)
        penalty=float((selected_penalties or {}).get(label,regularization))
        supported=int(counts.sum())>=50
        if (not supported or prior_mode is not None) and penalty not in pooled:
            pooled[penalty]=fit_feature_reward(expanded,all_counts,matrices,mask,feature_names=names,regularization=penalty,max_iter=max_iter)
        if supported:
            options={}
            if prior_mode is not None:
                if not pooled[penalty]['fit']['converged']:
                    raise ValueError(f'pooled feature reward prior at penalty {penalty} did not converge')
                center=np.asarray(pooled[penalty]['reward_weights'])
                options={'initial':center,'prior':center}
            fitted=fit_feature_reward(expanded,counts,matrices,mask,feature_names=names,regularization=penalty,max_iter=max_iter,**options)
        else:
            fitted=pooled[penalty]
        if not fitted['fit']['converged']: raise ValueError(f'linear actor {label} did not converge: {fitted["fit"]}')
        rewards.append(np.asarray(fitted['common_reward']));policies.append(fitted['policy']);values.append(fitted['value'])
        coefficients.append(np.asarray(fitted['reward_weights']).reshape(4,-1,80).transpose(1,0,2).tolist())
        offset,return_fit=_return_offset(activity.loc[activity.persona.eq(label)],penalty)
        offsets.append(offset);reports[label]={'action_records':int(counts.sum()),'action_users':int(panel['records'].loc[keep,'user_id'].nunique()),
            'source_prefix_users':int(prior[label]),'pooled_fallback':not supported,'regularization':penalty,'action_fit':fitted['fit'],'return_fit':return_fit}
    rewards=np.asarray(rewards);common=np.einsum('k,knj->nj',weights,rewards)
    posteriors={source_profile_id(contract['domain'],uid):np.eye(len(personas))[personas.index(label)].tolist()
        for uid,label in zip(prefix_users.user_id,prefix_users.cluster.astype(str))}
    result={'schema_version':1,'type':shopping.TYPE,'actions':list(shopping.ACTIONS),'discount':shopping.DISCOUNT,
        'regularization':float(regularization),'state_count':len(mask),'feature_names':shopping.FEATURE_NAMES,'context_keys':shopping.CONTEXT_KEYS,
        'encoder':encoder,'action_mask':mask.tolist(),'transition_csr':[shopping.pack_matrix(p) for p in matrices],
        'mixture_weights':weights.tolist(),'component_count':len(personas),'common_reward':common.tolist(),
        'component_reward_deviations':(rewards-common).tolist(),'component_offsets':np.zeros((len(personas),4)).tolist(),
        'component_policies':policies,'component_values':values,'profile_posteriors':posteriors,
        'persona_labels':personas,'source_cutoff_at':cutoff.isoformat(),'source_domain':contract['domain'],
        'linear_reward':{'feature_names':list(FEATURES),'state_basis':basis.tolist(),'state_phase_ids':phases.tolist(),
            'weights':coefficients,'observation_transform':transform,'coefficient_unit':'persona x private phase x4actions x80existing features',
            'no_added_public_or_missing_indicator_predictors':True},
        'return_activity':{'schema_version':1,'type':joint.ACTIVITY_TYPE,'horizon_days':30,'base_model_sha256':return_sha,
            'component_logit_offsets':offsets,'cutoff_at':cutoff.isoformat(),'fit_label_available_rule':'window_end < cutoff',
            'posterior_available_rule':'window_end < decision_clock'},
        'training':{'cutoff_at':cutoff.isoformat(),'inner_fit':inner_fit,'persona_models':reports,'split_audit':split,'dynamics_audit':dynamics},
        'posterior_bank_scope':'hash train users, frozen training-prefix BiGRU labels only; labels are descriptive conditioning in their own prefix',
        'limitations':['Finite-state mean reward basis approximates within-state variation;39public+existing40internal+intercept only.',
            'Prefix persona labels condition training descriptively; only labels already available at a future prediction may evaluate forecasting.',
            'Source fit and optimizer convergence do not establish free-generation fidelity or coupon response.']}
    if prior_mode is not None:
        result['linear_reward']['coefficient_prior']={
            'mode':prior_mode,'scope':'same source training partition, frozen encoder/state basis/dynamics',
            'rule':'L2(group weights - same-penalty pooled weights)/2; initialized at that pooled center',
            'applies_to':'four shopping feature-reward vectors; return-logit fit unchanged',
            'sparse_group_rule':'existing under50 action-record direct pooled fallback retained',
            'pooled_action_records':int(all_counts.sum()),
            'coefficient_order':'four canonical actions x private-phase blocks of existing80 features',
            'pooled_weights_by_penalty':{str(p):fit['reward_weights'] for p,fit in pooled.items()},
            'pooled_fit_by_penalty':{str(p):fit['fit'] for p,fit in pooled.items()},
            'interpretation':'empirical pooled-center regularization, not an independent prior or identified psychological preference'}
    validate_linear_actor(result)
    return result


def validate_linear_actor(block):
    linear=block.get('linear_reward')
    if linear is None: return
    if linear.get('feature_names')!=list(FEATURES): raise ValueError('linear reward predictor contract changed')
    basis=np.asarray(linear['state_basis'],float);phase=np.asarray(linear['state_phase_ids'],int)
    theta=np.asarray(linear['weights'],float);K=len(block['mixture_weights']);N=block['state_count']
    if basis.shape!=(N,80) or phase.shape!=(N,) or theta.shape!=(K,int(phase.max())+1,4,80) or not all(np.isfinite(x).all() for x in (basis,theta)):
        raise ValueError('invalid persona-phase linear coefficient representation')
    expected=np.stack([np.einsum('np,nap->na',basis,theta[k,phase]) for k in range(K)])
    compiled=np.asarray(block['common_reward'])[None]+np.asarray(block['component_reward_deviations'])
    if not np.allclose(expected,compiled,rtol=1e-10,atol=1e-10):
        raise ValueError('compiled actor reward differs from its linear feature weights')


def score_future_actions(block, data, assignments, *, start_at, end_at, heldout_inner=True):
    """Score known frozen labels and use the fixed prior for unseen-prefix users."""
    start, end = pd.Timestamp(start_at), pd.Timestamp(end_at)
    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError('future scoring needs explicit ordered aware boundaries')
    sessions = data['sessions']
    selected = (~sessions.invalid_total_cart & ~sessions.ambiguous_clock_sessions
                & ~sessions.is_censored & sessions.start_at.ge(start)
                & sessions.terminal_known_at.lt(end) & sessions.train_user)
    if heldout_inner:
        selected &= ~sessions.inner_user
    empty = lambda: {'records':0, 'users':0, 'customer_macro_nll':None}
    if not selected.any():
        return {key: empty() for key in [*block['persona_labels'], '__pooled__']}
    indices = record_indices(data, selected)
    records = data['records'].iloc[indices]
    labels = assignments.copy()
    labels['user_id'] = labels.user_id.astype(str)
    if labels.user_id.duplicated().any():
        raise ValueError('future scoring requires unique frozen persona users')
    labels = labels.set_index('user_id')
    timing = records.user_id.astype(str).map(labels.assignment_cutoff)
    if timing.notna().any() and timing[timing.notna()].gt(records.loc[timing.notna(), 'at']).any():
        raise ValueError('future policy evaluation cannot use unavailable persona labels')
    state, fallback = encode_records(block['encoder'], data['features'][indices], records)
    group = records.user_id.astype(str).map(labels.cluster.astype(str))
    target = records.action.to_numpy(int)
    mask = np.asarray(block['action_mask'],bool)
    if not mask[state,target].all():
        raise ValueError('future target lies outside the fitted observation support')
    result = {}
    policies = np.asarray(block['component_policies'])
    all_probability = policies[:,state,target]
    predicted = np.asarray(block['mixture_weights']) @ all_probability
    known = np.zeros(len(records),bool)
    for index,label in enumerate(block['persona_labels']):
        keep = group.eq(label).to_numpy()
        known |= keep
        if not keep.any():
            result[label] = empty()
            continue
        probability = all_probability[index,keep]
        predicted[keep] = probability
        if (probability<=0).any():
            raise ValueError('supported future action has zero learned probability')
        loss = -np.log(probability)
        users = records.user_id.to_numpy()[keep]
        result[label] = {'records':int(keep.sum()),'users':int(pd.Series(users).nunique()),
            'nll':float(loss.mean()),'customer_macro_nll':float(pd.Series(loss).groupby(users).mean().mean()),
            'encoder_fallback_records':int(fallback[keep].sum())}
    if (predicted<=0).any():
        raise ValueError('supported future action has zero fixed-prior mixture probability')
    losses = -np.log(predicted)
    users = records.user_id.to_numpy()
    result['__pooled__'] = {'records':len(records), 'users':int(pd.Series(users).nunique()),
        'nll':float(losses.mean()), 'customer_macro_nll':float(pd.Series(losses).groupby(users).mean().mean()),
        'unknown_prefix_users':int(pd.Series(users[~known]).nunique()),
        'unknown_prefix_records':int((~known).sum()),
        'unknown_prefix_rule':'fixed training persona mixture prior; no future labels or customer removal'}
    return result
