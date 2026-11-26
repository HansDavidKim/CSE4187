"""Empirical old-IRL actor to frozen BiGRU cluster transport, from training prefixes only."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from ..irl import fit_latent_shopping as shopping
from ..irl.source_contract import file_sha, utc_cutoff
from ..irl.source_personas import load_frozen_assignments

TYPE='training_prefix_soft_actor_cluster_contingency'


def soft_actor_cluster_mapping(model,features,records,assignments,cutoff_at):
    """Customer-weighted soft contingency; actor indices never stand for BiGRU labels."""
    block=model['latent_shopping'];cutoff=utc_cutoff(cutoff_at)
    if block['encoder'].get('lifecycle_memory') is not None:
        raise ValueError('this transport helper targets the original unphased Main IRL actor')
    labels=assignments.loc[assignments.split.ne('user_holdout')].copy()
    labels['user_id']=labels.user_id.astype(str)
    if labels.empty or labels.user_id.duplicated().any():raise ValueError('nonempty unique frozen training users required')
    if np.asarray(features).shape!=(len(records),len(shopping.FEATURE_NAMES)):
        raise ValueError('mapping requires exactly aligned original internal40 decisions')
    records=records.copy();records['user_id']=records.user_id.astype(str)
    keep=(records["at"].lt(cutoff)&records.target_at.lt(cutoff)&records.user_id.isin(labels.user_id)).to_numpy()
    selected=records.loc[keep];state,fallback=shopping.encode(block['encoder'],np.asarray(features)[keep])
    target=selected.action.to_numpy(int)
    policy=np.asarray(block['component_policies'],float)
    prior=np.asarray(block['mixture_weights'],float)
    if policy.ndim!=3 or policy.shape[0]!=len(prior) or policy.shape[2]!=5:
        raise ValueError('old Main actor policies must be K by state by5')
    lookup=pd.Index(labels.user_id);user=lookup.get_indexer(selected.user_id)
    likelihood=np.zeros((len(labels),len(prior)))
    probability=policy[:,state,target].T
    for component in range(len(prior)):
        np.add.at(likelihood[:,component],user,np.log(np.maximum(probability[:,component],1e-300)))
    posterior=likelihood+np.log(prior)
    posterior=np.exp(posterior-logsumexp(posterior,axis=1,keepdims=True))
    clusters=sorted(labels.cluster.astype(str).unique())
    counts=np.stack([posterior[labels.cluster.astype(str).eq(cluster)].sum(axis=0) for cluster in clusters],axis=1)
    if (counts.sum(axis=1)<=0).any():raise ValueError('every original actor needs positive frozen-cluster transport support')
    probabilities=counts/counts.sum(axis=1,keepdims=True)
    return {'schema_version':1,'type':TYPE,'cluster_labels':clusters,'probabilities':probabilities.tolist(),
        'soft_customer_counts':counts.tolist(),'training_users':len(labels),'known_decisions':len(selected),
        'zero_decision_users':int(len(labels)-selected.user_id.nunique()),
        'encoder_fallback_decisions':int(fallback.sum()),'zero_policy_entries':int((probability==0).sum()),
        'max_target_at':selected.target_at.max().isoformat() if len(selected) else None,'cutoff_at':cutoff.isoformat(),
        'weighting':'one posterior per outer-train frozen-prefix customer, then P(BiGRU cluster | old IRL actor)',
        'zero_policy_rule':'log probability floor1e-300 for finite empirical transport likelihood only; runtime policy unchanged',
        'modeling_scope':'private scenario transport; original actor and BiGRU cluster identities are distinct'}


def prepare_actor_cluster_mapping(base_model_path,decision_directory,assignment_path,*,prepared_source_path,
                                  persona_manifest_path,domain,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    base_path=Path(base_model_path);directory=Path(decision_directory);source=Path(prepared_source_path)
    manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
    source_sha=file_sha(source)
    if manifest['domain']!=domain or manifest['source_cache_sha256']!=source_sha:
        raise ValueError('mapping decisions must bind the matching prepared source')
    if manifest['core_feature_names']!=list(shopping.FEATURE_NAMES):raise ValueError('original internal40 order changed')
    for name,expected in manifest['files'].items():
        if file_sha(directory/name)!=expected:raise ValueError('mapping decision cache changed')
    descriptor=json.loads(source.with_suffix('.json').read_text());cutoff=descriptor['outer_cutoff_at']
    assignments,provenance=load_frozen_assignments(assignment_path,domain=domain,cutoff_at=cutoff,
        expected_sha256=file_sha(assignment_path),source_sha256=source_sha,persona_manifest_path=persona_manifest_path)
    records=pd.read_parquet(directory/'records.parquet')
    with np.load(directory/'decision_features.npz',allow_pickle=False) as cache:features=cache['core_features']
    model=json.loads(base_path.read_text())
    result=soft_actor_cluster_mapping(model,features,records,assignments,cutoff)
    result.update(base_model_sha256=file_sha(base_path),source_sha256=source_sha,domain=domain,
        decision_manifest_sha256=file_sha(manifest_path),decision_file_sha256=manifest['files'],
        assignments_sha256=file_sha(assignment_path),frozen_assignment_provenance=provenance)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result
