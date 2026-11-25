"""Representative IRL policy with personal multinomial-logit tastes and return.

The source-fitted representative critic stays frozen. Personal tastes change
immediate choice logits; they never trigger an individual Bellman replan.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from . import persona_reward_gmm as kernel
from .fit_gaussian_rewards import parameter_indices
from .source_contract import canonical_available_actions

TYPE = "source_persona_joint_choice_gmm"
BACKEND = "retail_choice_logit"
PARAMETERS = ["view_choice_logit", "cart_choice_logit", "remove_choice_logit",
              "purchase_choice_logit", "return_glm_logit"]
UNITS = ["choice_logit"] * 4 + ["return_glm_logit"]
INITIAL_SD = (.5, .5, .5, .5, .5)


def _representative(latent):
    probability=np.asarray(latent["component_policies"],float)
    mask=np.asarray(latent["action_mask"],bool)
    if (probability.ndim!=3 or probability.shape[2]!=5 or mask.shape!=probability.shape[1:]
            or not mask.any(axis=1).all() or not np.isfinite(probability).all()
            or (probability[:,mask]<=0).any() or (probability[:,~mask]!=0).any()
            or not np.allclose(probability.sum(axis=2),1.,atol=1e-10,rtol=0)):
        raise ValueError("choice tastes require normalized positive representative policies on their exact mask")
    available=canonical_available_actions(latent.get("encoder",{}).get("available_actions"))
    for action in ("view","cart","remove","purchase"):
        if action not in available and mask[:,("view","cart","remove","purchase").index(action)].any():
            raise ValueError("representative policy exposes an unsupported action")
    logs=np.zeros_like(probability);logs[:,mask]=np.log(probability[:,mask])
    return probability,mask,logs,parameter_indices(available)


def policy_log_provider(latent):
    """Return actor,canonical5 samples -> finite Q×(state*5) sparse-safe logs."""
    probability,mask,logs,active=_representative(latent)
    action_indices=[i for i in active if i<4]
    def provider(component_id,theta,*,temperature=1.):
        values=np.asarray(theta,float)
        if (isinstance(component_id,(bool,np.bool_)) or not isinstance(component_id,(int,np.integer))
                or not 0<=component_id<len(probability) or values.ndim!=2 or values.shape[1]!=5
                or not np.isfinite(values).all() or isinstance(temperature,(bool,np.bool_))
                or not np.isfinite(temperature) or temperature<=0):
            raise ValueError("choice policy needs a known actor, finite canonical5 tastes and positive temperature")
        result=[]
        for start in range(0,len(values),256):
            part=values[start:start+256];offset=np.zeros((len(part),5));offset[:,action_indices]=part[:,action_indices]
            score=(logs[int(component_id)][None]+offset[:,None,:])/temperature
            score=np.where(mask[None],score,-np.inf)
            normalized=score-logsumexp(score,axis=2,keepdims=True)
            result.append(np.where(mask[None],normalized,0.).reshape(len(part),-1))
        return np.concatenate(result) if result else np.empty((0,mask.size))
    return provider


def prepare_choice_policy(latent):
    representative,mask,_,active=_representative(latent);provider=policy_log_provider(latent)
    action_indices=[i for i in active if i<4]
    def solve(theta_action,*,component_id,temperature=1.):
        theta=np.asarray(theta_action,float)
        if theta.shape!=(4,) or not np.isfinite(theta).all():raise ValueError("choice offsets must be finite canonical4")
        if (isinstance(component_id,(bool,np.bool_)) or not isinstance(component_id,(int,np.integer))
                or not 0<=component_id<len(representative)):raise ValueError("invalid frozen representative actor")
        if isinstance(temperature,(bool,np.bool_)) or not np.isfinite(temperature) or temperature<=0:
            raise ValueError("choice temperature must be finite and positive")
        if temperature==1. and not theta[action_indices].any():return representative[int(component_id)].copy()
        sample=np.zeros((1,5));sample[0,:4]=theta
        logs=provider(component_id,sample,temperature=temperature).reshape(mask.shape)
        return np.where(mask,np.exp(logs),0.)
    return solve


def validate_choice_gmm(block,base_sha256,return_sha256,*,available_actions=None):
    if (not isinstance(block,dict) or type(block.get("schema_version")) is not int or block['schema_version']!=1
            or block.get('type')!=TYPE or block.get('backend')!=BACKEND or block.get('purpose')!='synthesis_only'
            or block.get('parameter_names')!=PARAMETERS or block.get('parameter_units')!=UNITS):
        raise ValueError("incompatible choice-logit GMM; reward offsets cannot be relabelled as choices")
    if block.get('base_model_sha256')!=base_sha256 or block.get('return_model_sha256')!=return_sha256:
        raise ValueError("choice GMM requires its exact representative actor and global return GLM")
    available=canonical_available_actions(block.get('available_actions'))
    if available_actions is not None and available!=canonical_available_actions(available_actions):
        raise ValueError("choice GMM observation support differs from the actor")
    active=parameter_indices(available);count=block.get('component_count');labels=block.get('persona_labels');indices=block.get('component_model_indices')
    if (type(count) is not int or count<2 or not isinstance(labels,list) or len(labels)!=count
            or any(not isinstance(x,str) or not x for x in labels) or len(set(labels))!=count
            or not isinstance(indices,list) or len(indices)!=count or len(set(indices))!=count
            or any(type(x) is not int or x<0 for x in indices)
            or block.get('fitted_parameter_names')!=[PARAMETERS[i] for i in active]):
        raise ValueError("choice GMM needs distinct known personas and canonical observed parameter support")
    conditional=block.get('condition_on_profile_components',False)
    if type(conditional) is not bool or conditional and sorted(indices)!=list(range(count)):
        raise ValueError("choice profile conditioning requires one-to-one representative actors")
    means,covariance,weights=(np.asarray(block.get(k),float) for k in ('means','covariances','mixture_weights'))
    if (means.shape!=(count,5) or covariance.shape!=(count,5,5) or weights.shape!=(count,)
            or not all(np.isfinite(x).all() for x in (means,covariance,weights)) or (weights<=0).any()
            or not np.isclose(weights.sum(),1.,atol=1e-10,rtol=0)
            or not np.allclose(covariance,covariance.swapaxes(1,2),atol=1e-12,rtol=1e-10)):
        raise ValueError("invalid choice Gaussian arrays or fixed persona probabilities")
    for name,positive in (('choice_temperatures',True),('dispersion_scales',False),
                          ('shopping_dispersion_scales',False),('return_dispersion_scales',False)):
        values=block.get(name,[1.]*count);numeric=np.asarray(values,float)
        if (numeric.shape!=(count,) or any(isinstance(x,(bool,np.bool_)) for x in values)
                or not np.isfinite(numeric).all() or (numeric<0).any() or positive and (numeric==0).any()):
            raise ValueError("invalid choice temperature or dispersion")
    try:return np.linalg.cholesky(covariance[:,active][:,:,active])
    except np.linalg.LinAlgError as error:raise ValueError("observed choice covariance must be positive definite") from error


def sample_choice(block,factors,solve_policy,profile_id,rng,*,profile_components=None):
    if block.get('type')!=TYPE or block.get('backend')!=BACKEND:
        raise ValueError("choice sampler requires its separate choice-logit artifact")
    weights=np.asarray(block['mixture_weights'],float);scope='choice_gmm_population_prior'
    if block.get('condition_on_profile_components',False):
        values=np.asarray(profile_components,float);indices=block['component_model_indices']
        if (values.shape!=weights.shape or not np.isfinite(values).all() or (values<0).any()
                or not np.isclose(values.sum(),1.,atol=1e-10,rtol=0) or sorted(indices)!=list(range(len(weights)))):
            raise ValueError("choice profile probabilities require the same one-to-one persona actors")
        weights=values[indices];scope='choice_gmm_profile_persona_prior'
    elif profile_components is not None:raise ValueError("unconditional choice GMM cannot use profile probabilities")
    component=int(rng.choice(len(weights),p=weights));active=parameter_indices(block.get('available_actions'))
    theta=np.zeros(5);scale=block.get('dispersion_scales',[1.]*len(weights))[component]
    shopping_scale=block.get('shopping_dispersion_scales',[scale]*len(weights))[component]
    return_scale=block.get('return_dispersion_scales',[scale]*len(weights))[component]
    scales=np.array([shopping_scale if i<4 else return_scale for i in active])
    theta[active]=np.asarray(block['means'][component])[active]
    if scales.any():
        theta[active]+=scales*(np.asarray(factors)[component]@rng.standard_normal(len(active)))
    policy=solve_policy(theta[:4],component_id=block['component_model_indices'][component],
        temperature=block.get('choice_temperatures',[1.]*len(weights))[component])
    return component,scope,{'action_offsets':theta[:4],'return_offset':float(theta[4]),'policy':policy}


def fit_choice_gmm(latent,actions,activity,assignments,*,component_model_indices,cutoff_at,seed=42,
                   max_updates=8,qmc_power=7,condition_on_profile_components=True,
                   base_model_sha256=None,return_model_sha256=None):
    """Fresh known-persona QMC fit; no reward-fit parameters are accepted."""
    provider=policy_log_provider(latent)
    result=kernel.fit_persona_reward_gmm(latent,actions,activity,assignments,
        component_model_indices=component_model_indices,cutoff_at=cutoff_at,seed=seed,
        max_updates=max_updates,qmc_power=qmc_power,condition_on_profile_components=condition_on_profile_components,
        base_model_sha256=base_model_sha256,return_model_sha256=return_model_sha256,
        policy_log_provider=provider,initial_sd=INITIAL_SD)
    old=kernel.PARAMETERS;rename=dict(zip(old,PARAMETERS))
    result.update(type=TYPE,backend=BACKEND,parameter_names=list(PARAMETERS),parameter_units=list(UNITS),
        available_actions=list(canonical_available_actions(latent.get('encoder',{}).get('available_actions'))),
        fitted_parameter_names=[PARAMETERS[i] for i in parameter_indices(latent.get('encoder',{}).get('available_actions'))])
    result['component_fitted_parameter_names']=[[rename[name] for name in row] for row in result['component_fitted_parameter_names']]
    for row in result['fit']['personas'].values():row['fitted_parameter_names']=[rename[name] for name in row['fitted_parameter_names']]
    result['fit'].update(algorithm='fresh known-persona Gaussian QMC EM on direct choice-logit and complete30d GLM likelihood',
        representative_critic='frozen source-fitted per-persona MaxEnt MDP policy; no individual Bellman replan',
        personal_choice_rule='softmax((log representative_probability + canonical action taste with END0) / T)',
        old_reward_fit_parameters_imported=False,parameter_units=list(UNITS))
    result['limitations'].append('This hybrid is representative IRL plus personal multinomial-logit taste; it is a different model from individual reward-space IRL replanning.')
    return result
