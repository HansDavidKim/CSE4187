"""Canonical bytes, private draws, backend units and zero-effect native paths."""
import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from simulate.v4.simulator import canonical_response as response
from simulate.v4.simulator.canonical_campaign import CanonicalMainCampaign,CanonicalTianchiDailyCampaign,make_campaign_env
from test_v4_main_campaign import factory as main_factory
from test_v4_tianchi_campaign import factory as daily_factory
from test_v4_tianchi_daily import source,model


def receipt_model():
    group={'receipts':1,'redeemed':1,'response_probability':1.,'redemption_delay_days':[0]}
    return {'schema_version':1,'type':'tianchi_o2o_descriptive_response','discount_rate':.1,'window_days':30,
        'inactivity_days':28,'groups':{k:group.copy() for k in ('pooled','no_observed_purchase','purchased_within_28d','lapsed_purchase')}}


def fixture_model(backend,bindings):
    matrix=np.zeros((49,4));matrix[0]=[-20.,20.,-20.,-20.]
    assignment={'types':list(response.TYPES),'matrix':matrix.tolist(),'latent_mean':[0.]*48,'latent_scale':[1.]*48,'group_logit_offsets':{}}
    original=json.dumps({'canonical_assignment':assignment},indent=3).encode();literal=response._matrix_literal(original)
    daily=backend=='tianchi_daily';roles=['T0','T1'] if daily else ['P1','P0']
    spec={'canonical_type_probabilities':dict(zip(response.TYPES,[.1,.4,.2,.3])),
          'purchase_logit_shift':.6,'return_logit_shift':.9}
    model={'schema_version':1,'type':response.TYPE,'backend':backend,'domain':'tianchi' if daily else 'cosmetics',
        'types':list(response.TYPES),'canonical_assignment':assignment,'personas':{r:copy.deepcopy(spec) for r in roles if r!='P1'},
        'default_strength':4.,'receipt_model':receipt_model(),'new_world_calibrated':False,
        'original_matrix_json_base64':base64.b64encode(literal).decode(),
        'original_matrix_json_sha256':hashlib.sha256(literal).hexdigest(),
        'original_matrix_float64_sha256':hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest(),
        'behavior_sha256':bindings,'entry_model_sha256':bindings.get('initial_entry_model_path')}
    bank=pd.DataFrame([{'user_id':str(100+i),'cluster':str(i),'persona':role,'source_profile_id':f'source{i}',
                       **{name:float(i)/10 for name in response.EMBEDDING}} for i,role in enumerate(roles)])
    return model,bank,original


def test_original_matrix_bytes_and_six_week_repeated_receipt_kernel():
    model,bank,_=fixture_model('retail_choice_logit',{'actor':'a'*64})
    adapter=response.CanonicalResponseAdapter(model,bank)
    changed=copy.deepcopy(model);changed['canonical_assignment']['matrix'][1][0]=1.
    with pytest.raises(ValueError,match='matrix bytes'):response.CanonicalResponseAdapter(changed,bank)
    positive={'purchase_offset':2.,'return_offset':3.,'strength':1.}
    for age,expected in [(0,1.),(7,.8),(14,.6),(21,.4),(28,.2),(35,0.),(42,0.)]:
        result=response.response_offsets(positive,[age]);assert result['purchase']==pytest.approx(2*expected)
    assert response.response_offsets(positive,[0,0])['purchase']==pytest.approx(2*np.sqrt(2))
    assert copy.deepcopy(adapter) is adapter


def test_unsupported_quiet_role_retained_with_original_matrix_type_and_zero_utility():
    model,bank,_=fixture_model('retail_choice_logit',{'actor':'a'*64});adapter=response.CanonicalResponseAdapter(model,bank)
    env=SimpleNamespace(model_shas={'actor':'a'*64},choice_space=True,latent={'persona_labels':['0','1']},component=0,
        source_profile_id='source0',response_rng=np.random.default_rng(3),state={},_receipts=[],_active=None)
    private=adapter.initialize(env)
    assert private['persona']=='P1' and not private['utility_supported'] and private['canonical_type']=='persuadable'
    at=pd.Timestamp('2024-01-01T00:00:00Z');env._receipts=[{'sent_at':at}]
    assert adapter.runtime_reward_offsets(env,at)=={'purchase_choice_logit_shift':0.,'return_logit_shift':0.}
    before=copy.deepcopy(env.response_rng.bit_generator.state);assert adapter.initialize(env) is private
    assert before==env.response_rng.bit_generator.state


def test_main_factory_zero_strength_native_paths_copy_and_rng(main_factory,tmp_path):
    native=main_factory(choice=True,gmm={},edit=lambda m:m['latent_shopping'].update(persona_labels=['0','1']),horizon_days=14)
    model,bank,_=fixture_model('retail_choice_logit',native.model_shas);adapter=response.CanonicalResponseAdapter(model,bank)
    bank_path=tmp_path/'bank.parquet';bank.to_parquet(bank_path,index=False)
    model.update(latent_bank_file='bank.parquet',latent_bank_sha256=response.file_sha(bank_path))
    path=tmp_path/'canonical.json';path.write_text(json.dumps(model))
    config=copy.deepcopy(native.config);config['source_simulation']['canonical_response_model_path']=str(path)
    env=make_campaign_env(config,domain='cosmetics',cost_model='D',warmup_days=30,min_history_days=0,horizon_days=14)
    adapter=env.canonical_adapter
    assert env.model_shas['canonical_response_model_path']==response.file_sha(path)
    assert env.model_shas['canonical_latent_bank']==response.file_sha(bank_path)
    native.reset(5);env.reset(5)
    for name in ('rng','visit_rng','review_rng'):
        assert getattr(native,name).bit_generator.state==getattr(env,name).bit_generator.state
    assert env.response['canonical_type']=='persuadable'
    original=env.response['latent48'].copy();adapter.set_strength(env,0.)
    control,treated=copy.deepcopy(env),copy.copy(env)
    assert control.response is not treated.response and not np.shares_memory(control.response['latent48'],treated.response['latent48'])
    control.advance(0,14);treated.advance(1,14)
    pd.testing.assert_frame_equal(control.tables['sessions'],treated.tables['sessions'])
    pd.testing.assert_frame_equal(control.tables['events'],treated.tables['events'])
    for name in ('rng','visit_rng','review_rng','response_rng'):
        assert getattr(control,name).bit_generator.state==getattr(treated,name).bit_generator.state
    np.testing.assert_array_equal(treated.response['latent48'],original)
    adapter.set_strength(treated,2.);at=treated.now
    offset=adapter.runtime_reward_offsets(treated,at)
    assert set(offset)=={'purchase_choice_logit_shift','return_logit_shift'}
    assert len(treated._features())==39


def test_daily_units_business_midnight_and_zero_strength_days(daily_factory):
    native=daily_factory(horizon_days=14)
    model,bank,_=fixture_model('tianchi_daily',native.model_shas);adapter=response.CanonicalResponseAdapter(model,bank)
    env=CanonicalTianchiDailyCampaign(native.config,canonical_adapter=adapter,warmup_days=30,min_history_days=0,horizon_days=14)
    env.reset(10);adapter.set_strength(env,0.)
    control,treated=copy.deepcopy(env),copy.deepcopy(env)
    control.advance(0,14);treated.advance(1,14)
    pd.testing.assert_frame_equal(control.tables['raw_daily'],treated.tables['raw_daily'])
    assert control.day_rng.bit_generator.state==treated.day_rng.bit_generator.state
    ready=treated._receipts[0]['response']['response_ready_at']
    assert ready.tz_convert('Asia/Seoul')==ready.tz_convert('Asia/Seoul').normalize()
    assert set(adapter.runtime_reward_offsets(treated,treated.now))=={'purchase_reward_shift','return_reward_shift'}
    assert 'orders' not in treated.records and len(treated._features())==39


def test_prepare_preserves_literal_bank_and_factory_rejects_unfitted_learning(tmp_path,monkeypatch):
    model,bank,original=fixture_model('retail_choice_logit',{'actor':'a'*64})
    template={'domain':'cosmetics','type':'v6persona_coupon_scenario','canonical_assignment':model['canonical_assignment'],
        'personas':model['personas'],'default_strength':4.,'response_kernel':'sqrt(sum(max(0,1-floor(age_days/7)/5)^2))'}
    canonical=tmp_path/'original.json';canonical.write_text(json.dumps(template,indent=3))
    prepared=tmp_path/'source.parquet';prepared.write_bytes(b'fixture-source')
    prepared.with_suffix('.json').write_text(json.dumps({'prepared_sha256':response.file_sha(prepared),
        'spec':{'domain':'cosmetics'},'outer_cutoff_at':'2020-01-29T00:00:00Z'}))
    frame=bank.drop(columns='source_profile_id').copy();frame['split']='train'
    monkeypatch.setattr(response,'load_frozen_assignments',lambda *args,**kwargs:(frame,{'scope':'fixture frozen48'}))
    assignment=tmp_path/'assignments_outer.parquet';assignment.write_bytes(b'frozen-assignment')
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps(receipt_model()))
    output=tmp_path/'adapter'
    built=response.prepare_response_assets(canonical,assignment,prepared_source_path=prepared,
        persona_manifest_path=tmp_path/'manifest.json',receipt_model_path=receipt,domain='cosmetics',backend='retail_choice_logit',
        behavior_sha256={'actor':'a'*64},output=output)
    assert (output/'canonical_original.json').read_bytes()==canonical.read_bytes()
    assert base64.b64decode(built['original_matrix_json_base64'])==response._matrix_literal(canonical.read_bytes())
    response.CanonicalResponseAdapter.load(output/'model.json')
    with pytest.raises(FileExistsError):
        response.prepare_response_assets(canonical,assignment,prepared_source_path=prepared,
            persona_manifest_path=tmp_path/'manifest.json',receipt_model_path=receipt,domain='cosmetics',backend='retail_choice_logit',
            behavior_sha256={'actor':'a'*64},output=output)
    with pytest.raises(ValueError,match='fresh-world coupon calibration'):
        make_campaign_env({'source_simulation':{}},canonical_model_path=output/'model.json',require_calibrated=True)
