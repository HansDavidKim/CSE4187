"""Frozen paper canonical types with backend-specific, uncalibrated utility hooks."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..irl.source_contract import file_sha, source_profile_id, utc_cutoff
from ..irl.source_personas import EMBEDDING, load_frozen_assignments
from .coupon_response import sample_coupon_response, validate_response_model

TYPES=("sure_thing","persuadable","sleeping_dog","lost_cause")
TYPE="frozen_canonical48_coupon_response_adapter"
BACKENDS=("retail_choice_logit","tianchi_daily")
STATE_KEY="canonical_coupon_response"


def response_offsets(response, receipt_ages_days, strength=1.):
    """Original v6 persona_coupon.response_offsets, unchanged numerical kernel."""
    ages = np.asarray(receipt_ages_days, dtype=float)
    calibrated = response.get("strength", 1.)
    if (not np.isfinite(ages).all() or (ages < 0).any() or not math.isfinite(strength) or strength < 0
            or not math.isfinite(calibrated) or calibrated < 0):
        raise ValueError("coupon strength and receipt ages must be finite and nonnegative")
    exposure = math.sqrt(float(np.square(np.maximum(0., 1 - np.floor(ages / 7) / 5)).sum()))
    exposure *= strength * calibrated
    return {"purchase": exposure * response["purchase_offset"],
            "return": exposure * response["return_offset"]}


def _matrix_literal(content):
    marker=content.index(b'"canonical_assignment"')
    start=content.index(b'[',content.index(b'"matrix"',marker))
    _,length=json.JSONDecoder().raw_decode(content[start:].decode('utf-8'))
    return content[start:start+length]


def _probabilities(values):
    values=np.asarray([values[k] for k in TYPES],float)
    if not np.isfinite(values).all() or (values<0).any() or not np.isclose(values.sum(),1.,atol=1e-10,rtol=0):
        raise ValueError('canonical type probabilities must be a finite four-type distribution')
    return values


def canonical_probabilities(model, embedding, role):
    assignment=model['canonical_assignment'];z=np.asarray(embedding,float)
    matrix=np.asarray(assignment['matrix'],dtype='<f8');mean=np.asarray(assignment['latent_mean'],float)
    scale=np.asarray(assignment['latent_scale'],float)
    if (matrix.shape!=(49,4) or z.shape[-1:]!=(48,) or mean.shape!=(48,) or scale.shape!=(48,)
            or not all(np.isfinite(x).all() for x in (matrix,z,mean,scale)) or (scale<=0).any()):
        raise ValueError('canonical response requires the original finite49x4 matrix and48D scaling')
    standardized=(z-mean)/scale
    x=np.concatenate([np.ones((*z.shape[:-1],1)),standardized],axis=-1)
    score=x@matrix
    offset=assignment.get('group_logit_offsets',{}).get(role)
    if offset is not None:
        offset=np.asarray(offset,float)
        if offset.shape!=(4,) or not np.isfinite(offset).all():raise ValueError('invalid canonical group intercept')
        score+=offset
    score-=score.max(axis=-1,keepdims=True)
    probability=np.exp(score)
    return probability/probability.sum(axis=-1,keepdims=True)


def validate_adapter(model,bank):
    if (type(model.get('schema_version')) is not int or model.get('schema_version')!=1 or model.get('type')!=TYPE or model.get('backend') not in BACKENDS
            or model.get('domain') not in ('cosmetics','electronics','tianchi') or model.get('types')!=list(TYPES)
            or model.get('new_world_calibrated') is not False and model.get('new_world_calibrated') is not True):
        raise ValueError('incompatible canonical response adapter')
    if (model['backend']=='tianchi_daily')!=(model['domain']=='tianchi'):
        raise ValueError('daily and retail coupon response units cannot be mixed')
    bindings=model.get('behavior_sha256')
    if (not isinstance(bindings,dict) or not bindings or any(not isinstance(k,str) or not isinstance(v,str)
            or len(v)!=64 or not set(v)<=set('0123456789abcdef') for k,v in bindings.items())):
        raise ValueError('canonical response requires exact newly bound behavior hashes')
    literal=base64.b64decode(model['original_matrix_json_base64'],validate=True)
    if model['canonical_assignment'].get('types')!=list(TYPES):
        raise ValueError('canonical matrix column order must retain the original four types')
    matrix=np.asarray(model['canonical_assignment']['matrix'],dtype='<f8')
    if (hashlib.sha256(literal).hexdigest()!=model['original_matrix_json_sha256']
            or np.asarray(json.loads(literal),dtype='<f8').tobytes()!=matrix.tobytes()
            or hashlib.sha256(matrix.tobytes()).hexdigest()!=model['original_matrix_float64_sha256']):
        raise ValueError('the original49x4 canonical matrix bytes changed')
    mapping=model.get('actor_cluster_mapping')
    if mapping is not None:
        probabilities=np.asarray(mapping['probabilities'],float)
        labels=mapping['cluster_labels']
        if (mapping.get('type')!='training_prefix_soft_actor_cluster_contingency'
                or mapping['base_model_sha256']!=model['behavior_sha256'].get('model_path')
                or len(set(labels))!=len(labels) or probabilities.ndim!=2 or probabilities.shape[1]!=len(labels)
                or not np.isfinite(probabilities).all() or (probabilities<0).any()
                or not np.allclose(probabilities.sum(axis=1),1.,rtol=0,atol=1e-10)
                or set(labels)!=set(bank.cluster.astype(str))):
            raise ValueError('original IRL actor to BiGRU cluster mapping must be a bound stochastic matrix')
    for role,specification in model['personas'].items():
        _probabilities(specification['canonical_type_probabilities'])
        for name in ('purchase_logit_shift','return_logit_shift'):
            value=specification[name]
            if isinstance(value,bool) or not math.isfinite(float(value)):
                raise ValueError('canonical purchase/return utility ratio must remain finite')
    validate_response_model(model['receipt_model'])
    required={'user_id','cluster','persona','source_profile_id',*EMBEDDING}
    if bank.empty or not required<=set(bank) or bank.user_id.duplicated().any() or not bank.user_id.map(lambda x:isinstance(x,str)).all():
        raise ValueError('a unique original-ID frozen48D training-prefix bank is required')
    if not np.isfinite(bank[list(EMBEDDING)].to_numpy(float)).all():
        raise ValueError('canonical latent bank must be finite48D')
    for role in bank.persona.unique():canonical_probabilities(model,np.zeros(48),str(role))
    for value in [model['default_strength'],*model.get('strength_by_persona',{}).values()]:
        if isinstance(value,bool) or not isinstance(value,(int,float,np.number)) or not math.isfinite(float(value)) or value<0:
            raise ValueError('new-world response strength must be finite and nonnegative')


def prepare_response_assets(canonical_path,assignment_path,*,prepared_source_path,persona_manifest_path,
                            receipt_model_path,domain,backend,behavior_sha256,output,entry_model_path=None,actor_cluster_mapping_path=None):
    """Publish new scenario assets; preserve original matrix bytes and never fit strengths."""
    canonical_path,source,output=Path(canonical_path),Path(prepared_source_path),Path(output)
    if output.exists():raise FileExistsError(output)
    original_bytes=canonical_path.read_bytes();original=json.loads(original_bytes)
    if original.get('domain')!=domain or original.get('type')!='v6persona_coupon_scenario':
        raise ValueError('canonical template must retain its original native domain')
    source_descriptor=json.loads(source.with_suffix('.json').read_text());source_sha=file_sha(source)
    if source_descriptor['prepared_sha256']!=source_sha or source_descriptor['spec']['domain']!=domain:
        raise ValueError('canonical bank requires its immutable prepared source')
    cutoff=source_descriptor['outer_cutoff_at']
    frame,provenance=load_frozen_assignments(assignment_path,domain=domain,cutoff_at=cutoff,
        expected_sha256=file_sha(assignment_path),source_sha256=source_sha,persona_manifest_path=persona_manifest_path)
    bank=frame.loc[frame.split.ne('user_holdout'),['user_id','cluster','persona',*EMBEDDING]].copy()
    bank['cluster']=bank.cluster.astype(str)
    bank['source_profile_id']=[source_profile_id(domain,uid) for uid in bank.user_id]
    receipt=json.loads(Path(receipt_model_path).read_text());literal=_matrix_literal(original_bytes)
    if not behavior_sha256 or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64 for k,v in behavior_sha256.items()):
        raise ValueError('new world behavior hashes are required before preparing coupon assets')
    if domain=='tianchi' and entry_model_path is None:
        raise ValueError('daily coupon bank must bind the new daily entry priors')
    matrix=np.asarray(original['canonical_assignment']['matrix'],dtype='<f8')
    model={'schema_version':1,'type':TYPE,'backend':backend,'domain':domain,'types':list(TYPES),
        'canonical_assignment':deepcopy(original['canonical_assignment']),'personas':deepcopy(original['personas']),
        'canonical_subgroups':deepcopy(original.get('canonical_subgroups',{})),
        'default_strength':original['default_strength'],'strength_by_persona':deepcopy(original.get('strength_by_persona',{})),
        'receipt_model':receipt,'response_kernel':original['response_kernel'],
        'behavior_sha256':dict(behavior_sha256),'source_sha256':source_sha,'cutoff_at':cutoff,
        'frozen_bank_provenance':provenance,'original_canonical_sha256':hashlib.sha256(original_bytes).hexdigest(),
        'original_matrix_json_base64':base64.b64encode(literal).decode(),
        'original_matrix_json_sha256':hashlib.sha256(literal).hexdigest(),
        'original_matrix_float64_sha256':hashlib.sha256(matrix.tobytes()).hexdigest(),
        'unsupported_role_rule':'original matrix type only; utility effect not identified and set to0; customer retained',
        'new_world_calibrated':False,'strength_status':'initializer_requires_fresh_world_calibration',
        'old_world_primary_failure_is_current_world_evidence':False,
        'parameter_units':({'purchase':'choice_logit','return':'retail30d_glm_logit'} if backend=='retail_choice_logit'
                           else {'purchase':'daily_joint_reward','return':'daily_activity_reward'}),
        'transport_assumption':'original purchase:return magnitude ratio fixed; new backend utility strength uncalibrated; daily return is not a transported30d GLM',
        'entry_model_sha256':file_sha(entry_model_path) if entry_model_path else None}
    if actor_cluster_mapping_path is not None:
        mapping=json.loads(Path(actor_cluster_mapping_path).read_text())
        if mapping.get('source_sha256')!=source_sha or mapping.get('cutoff_at')!=utc_cutoff(cutoff).isoformat():
            raise ValueError('actor cluster transport must use the same strict training prefix')
        model['actor_cluster_mapping']=mapping
        model['actor_cluster_mapping_sha256']=file_sha(actor_cluster_mapping_path)
    validate_adapter(model,bank)
    report={}
    for role,group in bank.groupby('persona',sort=True):
        probability=canonical_probabilities(model,group[list(EMBEDDING)].to_numpy(float),str(role)).mean(axis=0)
        report[str(role)]={'training_donors':len(group),'matrix_expected_type_proportions':dict(zip(TYPES,probability.tolist())),
            'original_target_type_proportions':model['personas'].get(str(role),{}).get('canonical_type_probabilities'),
            'utility_supported':str(role) in model['personas']}
    output.mkdir(parents=True,exist_ok=False)
    (output/'canonical_original.json').write_bytes(original_bytes)
    bank.to_parquet(output/'latent_bank.parquet',index=False)
    model['latent_bank_file']='latent_bank.parquet';model['latent_bank_sha256']=file_sha(output/'latent_bank.parquet')
    model['bank_expected_type_proportions']=report
    with (output/'model.json').open('x') as stream:json.dump(model,stream,indent=2,allow_nan=False);stream.write('\n')
    return model


class CanonicalResponseAdapter:
    """Immutable fitted assets; every customer draw and strength lives in env.state."""

    def __init__(self,model,bank):
        validate_adapter(model,bank)
        self._model=deepcopy(model);self._bank=bank.copy(deep=True)
        self._z=self._bank[list(EMBEDDING)].to_numpy(float,copy=True);self._z.setflags(write=False)
        self._by_profile={pid:i for i,pid in enumerate(self._bank.source_profile_id)}
        self._by_cluster={str(k):np.asarray(v,int) for k,v in self._bank.reset_index(drop=True).groupby('cluster').indices.items()}

    def __deepcopy__(self,memo):
        # Customer state/RNG is owned by the environment; these assets are never mutated.
        return self

    @classmethod
    def load(cls,path):
        path=Path(path);model=json.loads(path.read_text());bank_path=path.parent/model['latent_bank_file']
        if file_sha(bank_path)!=model['latent_bank_sha256']:raise ValueError('frozen canonical48D bank changed')
        result=cls(model,pd.read_parquet(bank_path))
        result._model_file_sha256=file_sha(path);result._bank_file_sha256=file_sha(bank_path)
        return result

    def bind(self,env):
        if any(env.model_shas.get(k)!=v for k,v in self._model['behavior_sha256'].items()):
            raise ValueError('coupon scenario requires its exact newly bound behavior world')
        daily=self._model['backend']=='tianchi_daily'
        native_daily=getattr(env,'settings',{}).get('backend')=='tianchi_daily_campaign'
        if daily!=native_daily or not daily and not getattr(env,'choice_space',False):
            raise ValueError('canonical coupon utility units differ from the native environment')
        if daily and env.model_shas.get('initial_entry_model_path')!=self._model['entry_model_sha256']:
            raise ValueError('canonical daily bank requires the exact new entry priors')

    def initialize(self,env):
        self.bind(env)
        if STATE_KEY in env.state:return env.state[STATE_KEY]
        mapping=self._model.get('actor_cluster_mapping')
        if self._model['backend']=='tianchi_daily':
            label=str(env.actor)
        elif mapping is not None:
            if env.component >= len(mapping['probabilities']):
                raise ValueError('original IRL actor exceeds its frozen BiGRU transport support')
            row=np.asarray(mapping['probabilities'][env.component],float)
            condition=getattr(env,'canonical_cluster_condition',None)
            if condition is not None:
                if condition not in mapping['cluster_labels'] or row[mapping['cluster_labels'].index(condition)]<=0:
                    raise ValueError('conditioned original actor has no requested BiGRU role support')
                label=condition
            else:
                label=mapping['cluster_labels'][int(env.response_rng.choice(len(row),p=row))]
        else:
            label=str(env.latent['persona_labels'][env.component])
        candidates=self._by_cluster.get(label)
        if candidates is None or not len(candidates):raise ValueError('native actor has no frozen48D response donors')
        matched=self._by_profile.get(env.source_profile_id)
        if matched is not None and str(self._bank.iloc[matched].cluster)==label:
            index=matched;scope='same_source_profile_namespace'
        else:
            index=int(candidates[int(env.response_rng.integers(len(candidates)))]);scope='same_actor_training_prefix_bank'
        row=self._bank.iloc[index];role=str(row.persona)
        probabilities=canonical_probabilities(self._model,self._z[index],role)
        kind=TYPES[int(env.response_rng.choice(4,p=probabilities))]
        sign=1 if kind=='persuadable' else -1 if kind=='sleeping_dog' else 0
        specification=self._model['personas'].get(role)
        response={'persona':role,'actor_label':label,'canonical_type':kind,'latent_donor_id':str(row.user_id),
            'latent48':self._z[index].copy(),'canonical_probabilities':probabilities.copy(),'assignment_scope':scope,
            'utility_supported':specification is not None,
            'actor_cluster_scope':'soft_training_prefix_contingency' if mapping is not None else 'same_frozen_persona_actor',
            'source_actor_index':int(env.component) if self._model['backend']!='tianchi_daily' else int(env.actor),
            'purchase_offset':sign*specification['purchase_logit_shift'] if specification else 0.,
            'return_offset':sign*specification['return_logit_shift'] if specification else 0.,
            'strength':float(self._model.get('strength_by_persona',{}).get(role,self._model['default_strength']))}
        env.state[STATE_KEY]=response
        return response

    def set_strength(self,env,strength):
        if isinstance(strength,bool) or not isinstance(strength,(int,float,np.number)) or not math.isfinite(float(strength)) or strength<0:
            raise ValueError('scenario strength must be finite and nonnegative')
        self.initialize(env)['strength']=float(strength)

    def response_sampler(self,env,sent_at):
        self.initialize(env);sent_at=utc_cutoff(sent_at)
        if self._model['backend']=='tianchi_daily':
            known=[r for r in env.records['raw_daily'] if r['purchase'] and r['known_at']<=sent_at]
            last=max((r['day_start'] for r in known),default=None)
        else:last=env.state['last_purchase']
        response=sample_coupon_response(self._model['receipt_model'],env.seed,env.uid,sent_at,last)
        response['response_selected']=bool(response['response_selected'])
        if not response['response_selected']:response['response_ready_at']=None
        return response

    def runtime_reward_offsets(self,env,decision_at):
        daily=self._model['backend']=='tianchi_daily'
        zero=({'purchase_reward_shift':0.,'return_reward_shift':0.} if daily
              else {'purchase_choice_logit_shift':0.,'return_logit_shift':0.})
        at=utc_cutoff(decision_at)
        ages=[(at-r['sent_at']).total_seconds()/86400 for r in env._receipts if r['sent_at']<=at]
        if not ages:return zero
        response=self.initialize(env)
        if not response['strength']:return zero
        offsets=response_offsets(response,ages)
        if not daily and getattr(env,'_active',None) is not None and env._active.get('converted'):
            offsets['purchase']=0.
        return ({'purchase_reward_shift':offsets['purchase'],'return_reward_shift':offsets['return']} if daily else
                {'purchase_choice_logit_shift':offsets['purchase'],'return_logit_shift':offsets['return']})
