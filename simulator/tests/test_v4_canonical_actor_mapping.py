"""Old IRL actors are transported through a soft prefix contingency, never index labels."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from simulate.v4.simulator import canonical_actor_mapping as transport
from simulate.v4.simulator import canonical_response as response
from test_v4_canonical_response import fixture_model


def test_customer_soft_contingency_uses_only_known_train_prefix(monkeypatch):
    monkeypatch.setattr(transport.shopping,'encode',lambda enc,x:(np.zeros(len(x),int),np.zeros(len(x),bool)))
    model={'latent_shopping':{'encoder':{},'mixture_weights':[.5,.5],
        'component_policies':[[[.8,0,0,.1,.1]],[[.1,0,0,.8,.1]]]}}
    assignments=pd.DataFrame({'user_id':['a','b','z','c'],'cluster':['9','2','9','2'],
        'split':['train','train','user_holdout','train']})
    before=pd.Timestamp('2020-01-01T00:00:00Z');cutoff=before+pd.Timedelta(days=1)
    records=pd.DataFrame({'user_id':['a','a','b','b','z','a'],
        'at':[before]*5+[cutoff], 'target_at':[before]*5+[cutoff], 'action':[0,0,3,3,0,3]})
    result=transport.soft_actor_cluster_mapping(model,np.zeros((6,40)),records,assignments,cutoff)
    assert result['training_users']==3 and result['known_decisions']==4 and result['zero_decision_users']==1
    assert result['cluster_labels']==['2','9']
    expected=np.array([[1/65+.5,64/65],[64/65+.5,1/65]])
    np.testing.assert_allclose(result['soft_customer_counts'],expected)
    np.testing.assert_allclose(result['probabilities'],expected/expected.sum(axis=1,keepdims=True))
    records.loc[records.user_id.eq('z') | records['at'].ge(cutoff),'action']=4
    assert transport.soft_actor_cluster_mapping(model,np.zeros((6,40)),records,assignments,cutoff)==result


def test_private_response_uses_transport_instead_of_old_actor_index():
    model,bank,_=fixture_model('retail_choice_logit',{'model_path':'a'*64})
    model['actor_cluster_mapping']={'type':transport.TYPE,'base_model_sha256':'a'*64,
        'cluster_labels':['0','1'],'probabilities':[[0.,1.],[1.,0.]]}
    adapter=response.CanonicalResponseAdapter(model,bank)
    env=SimpleNamespace(model_shas={'model_path':'a'*64},choice_space=True,latent={},component=0,
        source_profile_id='SYN000000000',response_rng=np.random.default_rng(4),state={})
    private=adapter.initialize(env)
    assert private['actor_label']=='1' and private['source_actor_index']==0 and private['persona']=='P0'
    assert private['utility_supported'] and private['actor_cluster_scope']=='soft_training_prefix_contingency'
    saved=copy.deepcopy(env.response_rng.bit_generator.state)
    assert adapter.initialize(env) is private and saved==env.response_rng.bit_generator.state
    changed=copy.deepcopy(model);changed['actor_cluster_mapping']['base_model_sha256']='b'*64
    with pytest.raises(ValueError,match='bound stochastic matrix'):response.CanonicalResponseAdapter(changed,bank)
