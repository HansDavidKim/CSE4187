"""Source START-conditioned history is late-entry past only, with unchanged future gates."""
import copy,json
import numpy as np
import pandas as pd

from simulate.v4.simulator import main_campaign as main,source_history
from test_v4_main_campaign import factory


def test_history_past_clocks_preserve_calendar_and_do_not_force_future(factory,monkeypatch,tmp_path):
    native=factory(warmup_days=364,horizon_days=14)
    path=tmp_path/'history.json';path.write_text('{}')
    config=copy.deepcopy(native.config);config['source_simulation']['source_history_schedule_path']=str(path)
    returns=json.loads(open(config['source_simulation']['return_model_path']).read())
    returns.update(intercept=-20.,cold_start_probability=0.)
    open(config['source_simulation']['return_model_path'],'w').write(json.dumps(returns))
    bank={'records':[{'profile_id':'p0'}]}
    monkeypatch.setattr(source_history,'load_history_schedule',lambda *args,**kwargs:bank)
    monkeypatch.setattr(source_history,'draw_history_schedule',lambda bank,rng,at,earliest:
        ('p0',[at-pd.Timedelta(days=100),at-pd.Timedelta(days=3)]))
    monkeypatch.setattr(source_history,'history_provenance',lambda *args:{'past_actions':'fresh IRL','future':'unchanged gate'})
    env=main.MainCampaignEnv(config,warmup_days=364,horizon_days=14,min_history_days=91,feature_builder=lambda _:np.zeros(39))
    env.reset(12)
    decision=env.origin_clock+pd.Timedelta(days=364)
    assert env.decision_start==decision and env.origin==decision-pd.Timedelta(days=100)
    assert env.records['customers'][0]['signed_up_at']==env.origin
    assert [r['started_at'] for r in env.records['sessions']]==[decision-pd.Timedelta(days=100),decision-pd.Timedelta(days=3)]
    assert env.records['_metadata']['initial_candidate_attempts']==1
    assert env.records['_metadata']['calendar_warmup_start_at']==env.origin_clock.isoformat()
    assert env.state['first_event_at']==env.origin
    before=len(env.records['sessions']);theta=env.theta.copy()
    env.advance(0,14)
    assert len(env.records['sessions'])==before # actual future30d return gate is near zero
    np.testing.assert_array_equal(env.theta,theta)
    assert not env._window['entry'] and env._window['lower']==decision
