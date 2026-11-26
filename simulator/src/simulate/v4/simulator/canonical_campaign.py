"""Construct native Main-choice or Tianchi-daily campaigns with canonical hooks."""
from pathlib import Path
import numpy as np

from ..configuration import load_config
from .canonical_response import CanonicalResponseAdapter,STATE_KEY
from .main_campaign import MainCampaignEnv
from .tianchi_campaign import TianchiDailyCampaignEnv


class _CanonicalCampaign:
    def __init__(self,config,*,canonical_adapter,calibration_persona=None,**kwargs):
        if 'response_sampler' in kwargs or 'runtime_reward_offsets' in kwargs:
            raise ValueError('canonical factory owns both response hooks')
        self.canonical_adapter=canonical_adapter
        self.canonical_cluster_condition=None
        self.calibration_persona=calibration_persona
        if calibration_persona is not None:
            model=canonical_adapter._model
            clusters=canonical_adapter._bank.loc[canonical_adapter._bank.persona.eq(calibration_persona),'cluster'].astype(str).unique()
            if len(clusters)!=1:raise ValueError('calibration role requires one frozen BiGRU cluster')
            self.canonical_cluster_condition=str(clusters[0])
            if model['backend']=='retail_choice_logit':
                mapping=model.get('actor_cluster_mapping')
                if mapping is not None:
                    column=mapping['cluster_labels'].index(self.canonical_cluster_condition)
                    kwargs['component_conditioning']=[row[column] for row in mapping['probabilities']]
        super().__init__(config,response_sampler=canonical_adapter.response_sampler,
            runtime_reward_offsets=canonical_adapter.runtime_reward_offsets,**kwargs)
        canonical_adapter.bind(self)
        if (calibration_persona is not None and canonical_adapter._model['backend']=='retail_choice_logit'
                and canonical_adapter._model.get('actor_cluster_mapping') is None):
            if self.history_bank is None:
                raise ValueError('source-persona calibration requires a bound observed-history donor bank')
            selected=[]
            for row in self.history_bank['records']:
                profile=self._history_profiles[row['profile_id']]
                prior=np.asarray(profile.get('component_probabilities',self.latent['profile_posteriors'].get(
                    row['profile_id'],self.latent['mixture_weights'])),float)
                if not np.isin(prior,[0.,1.]).all() or prior.sum()!=1:
                    raise ValueError('direct source-persona conditioning requires exact frozen one-hot actor identity')
                actor_label=str(self.latent['persona_labels'][int(prior.argmax())])
                if actor_label==self.canonical_cluster_condition:selected.append(row)
            if not selected:raise ValueError('requested response role has no eligible original source-history support')
            self.history_bank={**self.history_bank,'records':selected}
        if calibration_persona is not None and canonical_adapter._model['backend']=='tianchi_daily':
            self.profile_probabilities=(np.asarray(self.labels)==self.canonical_cluster_condition).astype(float)
            if self.profile_probabilities.sum()!=1:raise ValueError('daily calibration role lacks its frozen source actor')

    def reset(self,seed):
        observation=super().reset(seed)
        self.canonical_adapter.initialize(self)
        model=self.canonical_adapter._model
        self.records['_metadata']['canonical_response']={
            'type':model['type'],'backend':model['backend'],'original_canonical_sha256':model.get('original_canonical_sha256'),
            'matrix_float64_sha256':model['original_matrix_float64_sha256'],
            'new_world_calibrated':model['new_world_calibrated'],'parameter_units':model.get('parameter_units'),
            'unsupported_role':'retained; original matrix type, utility not identified=0',
            'latent_visibility':'private env.state only; no new public feature',
            'calibration_persona':self.calibration_persona}
        return observation

    @property
    def response(self):
        return self.state[STATE_KEY]


class CanonicalMainCampaign(_CanonicalCampaign,MainCampaignEnv):
    pass


class CanonicalTianchiDailyCampaign(_CanonicalCampaign,TianchiDailyCampaignEnv):
    pass


def make_campaign_env(config,*,canonical_model_path=None,domain=None,cost_model='S',require_calibrated=False,**kwargs):
    parsed=load_config(config,assets_root=kwargs.get('assets_root')) if isinstance(config,(str,Path)) else config
    canonical_model_path=canonical_model_path or parsed['source_simulation'].get('canonical_response_model_path')
    if canonical_model_path is None:raise ValueError('an explicitly prepared canonical response model is required')
    canonical_model_path=Path(canonical_model_path)
    if not canonical_model_path.is_absolute():
        root=Path(kwargs.get('assets_root') or (Path(config).resolve().parent if isinstance(config,(str,Path)) else Path.cwd()))
        canonical_model_path=root/canonical_model_path
    adapter=CanonicalResponseAdapter.load(canonical_model_path)
    if domain is not None and domain!=adapter._model['domain']:
        raise ValueError('requested campaign domain differs from the canonical bank')
    if require_calibrated and adapter._model['new_world_calibrated'] is not True:
        raise ValueError('fresh-world coupon calibration is required before benchmark learning')
    cls=CanonicalTianchiDailyCampaign if adapter._model['backend']=='tianchi_daily' else CanonicalMainCampaign
    env=cls(config,canonical_adapter=adapter,cost_model=cost_model,**kwargs)
    env.model_shas['canonical_response_model_path']=adapter._model_file_sha256
    env.model_shas['canonical_latent_bank']=adapter._bank_file_sha256
    return env


make_canonical_campaign_env=make_campaign_env
