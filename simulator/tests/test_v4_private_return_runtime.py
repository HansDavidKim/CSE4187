"""Delayed observed completion, persistent private eta, and optional runtime wiring."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl import return_state
from simulate.v4.simulator.private_return import PrivateReturnState
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.raw_store import RAW_COLUMNS
from test_v4_checkout_simulator import checkout_fixture, _customers
from test_v4_gaussian_simulator import gaussian_fixture, enable_gaussian


def test_completion_waits_for_strict_known_clock_and_never_updates_censoring():
    start=pd.Timestamp('2020-01-01T00:00Z');rng=np.random.default_rng(4)
    state=PrivateReturnState(return_state.make_model(.5,.2),rng)
    assert state.value_at(start)==0. and state.eta is None
    first=state.begin_session(start)
    end=start+pd.Timedelta(seconds=10)
    state.complete_session(end,'causal_policy_end',False)
    state.complete_session(end,'max_events',True)
    state.complete_session(end,'window_end',True)
    known=end+pd.Timedelta(minutes=30)
    assert state.value_at(known)==first and state.updates==0
    state.value_at(known+pd.Timedelta(nanoseconds=1))
    assert state.updates==1 and not state.pending
    persistent=state.eta;before=copy.deepcopy(rng.bit_generator.state)
    assert state.begin_session(start+pd.Timedelta(days=60))==persistent
    assert state.updates==1 and rng.bit_generator.state==before
    with pytest.raises(ValueError,match='backwards'):
        state.value_at(start)


def test_zero_sigma_initialization_and_session_updates_consume_no_rng():
    start=pd.Timestamp('2020-01-01T00:00Z');rng=np.random.default_rng(2)
    before=copy.deepcopy(rng.bit_generator.state)
    state=PrivateReturnState(return_state.make_model(.9,0.),rng)
    assert state.begin_session(start)==0.
    state.complete_session(start,'donor_length',False)
    assert state.value_at(start+pd.Timedelta(hours=1))==0.
    assert state.updates==1 and rng.bit_generator.state==before


def test_optional_zero_sigma_preserves_business_records_and_dynamic_counts(gaussian_fixture,tmp_path):
    config,gaussian,path=gaussian_fixture
    configured=enable_gaussian(config,gaussian,path)
    baseline=EcommerceSimulator(configured).generate(months=3,seed=17,customers=_customers())
    state_path=tmp_path/'state.json';state_path.write_text(json.dumps(return_state.make_model(.5,0.)))
    configured=copy.deepcopy(configured);configured['source_simulation']['return_state_model_path']=str(state_path)
    current=EcommerceSimulator(configured).generate(months=3,seed=17,customers=_customers())
    for name in (*RAW_COLUMNS,'treatments'):
        pd.testing.assert_frame_equal(baseline.tables[name],current.tables[name])
    np.testing.assert_array_equal(baseline.tables['diagnostics'].no_coupon_return_probability_30d,
                                  current.tables['diagnostics'].no_coupon_return_probability_30d)
    assert current.tables['diagnostics'].return_eta_at_snapshot.eq(0.).all()
    detail=current.tables['session_diagnostics'];end=pd.Timestamp(current.metadata['end_at'])
    complete=detail.loc[detail.termination_reason.isin(['causal_policy_end','donor_length']) & ~detail.truncated_session]
    known=complete.loc[(complete.as_of_at+pd.Timedelta(minutes=30)).lt(end)].groupby('customer_id').size()
    actual=current.tables['diagnostics'].set_index('customer_id').return_eta_known_completed_sessions
    assert actual.to_dict()==known.reindex(actual.index,fill_value=0).to_dict()
    for uid,group in detail.groupby('customer_id'):
        observed=complete.loc[complete.customer_id.eq(uid),'as_of_at']+pd.Timedelta(minutes=30)
        expected=[int(observed.lt(at).sum()) for at in group.session_started_at]
        assert group.return_eta_known_updates_at_start.tolist()==expected


def test_runtime_adds_global_static_and_eta_once(gaussian_fixture,tmp_path,monkeypatch):
    from simulate.v4.irl import fit_return
    config,gaussian,path=gaussian_fixture;config=enable_gaussian(config,gaussian,path)
    config['source_simulation']['retention_mode']='source'
    state_path=tmp_path/'state.json';state_path.write_text(json.dumps(return_state.make_model(.5,.2)))
    config['source_simulation']['return_state_model_path']=str(state_path)
    monkeypatch.setattr(fit_return,'return_probability',lambda model,context:.25)
    result=EcommerceSimulator(config).generate(months=3,seed=17,customers=_customers())
    detail=result.tables['diagnostics']
    logit=np.log(.25/.75)+detail.gaussian_return_logit_offset+detail.return_eta_at_snapshot
    np.testing.assert_allclose(detail.no_coupon_return_probability_30d,1/(1+np.exp(-logit)),rtol=0,atol=1e-12)
    assert detail.return_eta_at_snapshot.ne(0.).any()
