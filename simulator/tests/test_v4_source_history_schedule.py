"""Strict-prefix initialization reuses clocks, never heldout/future actions."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl.source_contract import source_profile_id
from simulate.v4.simulator import source_history
from simulate.v4.simulator.engine import EcommerceSimulator
from test_v4_checkout_simulator import checkout_fixture


def bank_fixture(tmp_path, model):
    uid = next(str(i) for i in range(100) if hashlib.sha256(f"42:{i}".encode()).digest()[0] < 205)
    cache = tmp_path/'prepared.parquet'
    cache.write_bytes(b'bound prepared source fixture')
    model['summary_scope'] = {'mode':'train_only'}
    model['source'].update(domain='electronics', training_cutoff_at='2021-02-01T00:00:00Z',
        prepared_source_cache={'path':str(cache),'sha256':source_history.file_sha(cache)})
    profile = copy.deepcopy(model['profiles'][0])
    profile['profile_id'] = source_profile_id('electronics',uid)
    model['profiles'] = [profile]
    bank = {'schema_version':1,'type':source_history.TYPE,'domain':'electronics',
        'source_cutoff_at':'2021-02-01T00:00:00Z','prepared_source_sha256':source_history.file_sha(cache),
        'conditioning':{'history_days':0,'recent_session_days':28},
        'records':[{'source_user_id':uid,'profile_id':profile['profile_id'],
                    'session_start_offsets_seconds':[-25*86400.,-2*86400.]}]}
    path=tmp_path/'bank.json';path.write_text(json.dumps(bank))
    return path,bank


@pytest.mark.parametrize('case', ['heldout','future','wrong_source','wrong_cutoff','wrong_eligibility'])
def test_bank_rejects_heldout_future_or_unbound_clocks(checkout_fixture,tmp_path,case):
    _,model=copy.deepcopy(checkout_fixture);path,bank=bank_fixture(tmp_path,model)
    if case=='heldout':
        uid=next(str(i) for i in range(100) if hashlib.sha256(f"42:{i}".encode()).digest()[0]>=205)
        bank['records'][0]['source_user_id']=uid
    elif case=='future':bank['records'][0]['session_start_offsets_seconds'][-1]=0.
    elif case=='wrong_source':bank['prepared_source_sha256']='other'
    elif case=='wrong_cutoff':bank['source_cutoff_at']='2021-02-02T00:00:00Z'
    else:bank['conditioning']['recent_session_days']=30
    path.write_text(json.dumps(bank))
    with pytest.raises(ValueError):source_history.load_history_schedule(path,model,history_days=0,eligibility_days=28)


def test_schedule_translation_preserves_gaps_and_cannot_extend_observation(checkout_fixture,tmp_path):
    _,model=copy.deepcopy(checkout_fixture);path,_=bank_fixture(tmp_path,model)
    bank=source_history.load_history_schedule(path,model,history_days=0,eligibility_days=28)
    t=pd.Timestamp('2024-02-05T00:00:00Z')
    _,starts=source_history.draw_history_schedule(bank,np.random.default_rng(1),t,earliest=t-pd.Timedelta(days=30))
    assert starts==[t-pd.Timedelta(days=25),t-pd.Timedelta(days=2)]
    with pytest.raises(ValueError,match='available past'):source_history.draw_history_schedule(bank,np.random.default_rng(1),t,earliest=t-pd.Timedelta(days=20))


def test_engine_generates_past_actions_at_bound_starts_and_does_not_force_return(checkout_fixture,tmp_path,monkeypatch):
    from simulate.v4.irl import fit_latent_shopping as latent,fit_visit_timing as timing
    config,model=copy.deepcopy(checkout_fixture);path,bank=bank_fixture(tmp_path,model)
    model['latent_shopping']={'type':'fixture','mixture_weights':[1.], 'discount':.95,
        'regularization':.01,'state_count':1,'feature_names':[]}
    actor=tmp_path/'actor.json';actor.write_text(json.dumps(model))
    clock=tmp_path/'clock.json';clock.write_text(json.dumps({'source':model['source'],'type':'fixture'}))
    head=Path(config['source_simulation']['return_model_path'])
    return_model=json.loads(head.read_text());return_model.update(intercept=-100.,cold_start_probability=0.)
    target=tmp_path/'return.json';target.write_text(json.dumps(return_model))
    config['source_simulation'].update(model_path=str(actor),source_history_schedule_path=str(path),
        visit_timing_model_path=str(clock),return_model_path=str(target),retention_mode='source',
        fail_on_session_cap=True,min_customer_history_days=0,eligibility_window_days=28)
    monkeypatch.setattr(latent,'validate_model',lambda m:None)
    monkeypatch.setattr(latent,'sample_component',lambda *args:(0,'fixture'))
    monkeypatch.setattr(latent,'policy',lambda m,c,z:{a:float(a==('view' if c['session_decision_count']==0 else 'end')) for a in ['view','cart','remove','purchase','end']})
    monkeypatch.setattr(timing,'validate_model',lambda m:None)
    monkeypatch.setattr(timing,'sample_session_offsets',lambda *args:([0.],'fixture'))
    monkeypatch.setattr(timing,'first_session_share',lambda *args:1.)
    result=EcommerceSimulator(config).generate(n_customers=2,months=3,seed=17)
    t=pd.Timestamp(result.metadata['as_of_at'])
    for _,sessions in result.tables['sessions'].groupby('customer_id'):
        assert sessions.started_at.tolist()==[t-pd.Timedelta(days=25),t-pd.Timedelta(days=2)]
    events=result.tables['events'];assert events.event_type.eq('view').all()
    assert len(events)==4 and (events.event_at<t).all()
    assert result.metadata['candidate_customers']==2
    assert result.metadata['source_history_conditioning']['donor_users']==1
