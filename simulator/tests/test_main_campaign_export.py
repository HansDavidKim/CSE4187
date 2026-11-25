"""Factual exports keep daily missingness and recompute preassignment states."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from simulate.v4.data.campaign_export import export_campaign,load_campaign_inputs
from simulate.v4.features import EXPANDED_FEATURES
from simulate.v4.irl.tianchi_daily import public_observation
from simulate.v4.simulator.tianchi_campaign import model_date
from simulate.v4.time_spec import weekly_calendar,weekly_metadata


class DailyFixture:
    warmup_days=horizon_days=364
    model_shas={'fixture':'a'*64}
    canonical_adapter=SimpleNamespace(_model={'new_world_calibrated':True,'domain':'tianchi','engineering_fixture_only':'unit fixture'})
    customer_id_for_seed=staticmethod(str)

    def reset(self,seed):
        self.origin,self.now,self.end=weekly_calendar('2024-01-01T00:00:00+09:00')
        self.uid=str(seed);self.t0=self.now;self.issued=[]
        self.records={'_metadata':{**weekly_metadata(),'initial_candidate_attempts':1}}
        dates=pd.date_range(self.origin,self.end,freq='D',inclusive='left')
        index=np.arange(len(dates))
        self.days=pd.DataFrame({'customer_id':self.uid,'day_start':dates,'known_at':dates+pd.Timedelta(days=1),
            'view':index%5==0,'purchase':index%11==0})
        history=pd.DataFrame({'user_id':self.uid,'day_start':self.days.day_start.map(model_date),
            'view':self.days.view,'purchase':self.days.purchase})
        observed=public_observation(history,model_date(self.now),source_contract={'feature_windows_days':[28,56,91]})
        values=np.asarray([np.nan if observed[k] is None else observed[k] for k in EXPANDED_FEATURES])
        if getattr(self,'corrupt_online',False):values[EXPANDED_FEATURES.index('session_count_30d')]+=1
        return values

    def advance(self,action,days=7):
        if action:
            self.issued.append(dict(customer_id=self.uid,coupon_id=f'{self.uid}:C1',sent_at=self.now,
                expires_at=self.now+pd.Timedelta(days=28),send_cost_minor=100,discount_minor=None))
        self.now+=pd.Timedelta(days=days)
        return None,0.,self.now==self.end,{}

    @property
    def tables(self):
        return {'customers':pd.DataFrame({'customer_id':[self.uid]}),'raw_daily':self.days.copy(),
            'coupons':pd.DataFrame(self.issued),'coupon_redemptions':pd.DataFrame(),
            '_metadata':deepcopy(self.records['_metadata'])}


def test_daily_export_has_raw_recomputed39_factual_targets_and_null_cash(tmp_path):
    output=tmp_path/'daily'
    env=DailyFixture();env.canonical_adapter=deepcopy(env.canonical_adapter)
    env.canonical_adapter._model['engineering_fixture_only']='unit fixture'
    result=export_campaign(env,output,domain='tianchi',customers=2,seed=42)
    assert result['snapshot_raw_recalculation_matches'] and len(result['customer_runtime'])==2
    assert result['production_export'] is False
    features=pd.read_parquet(output/'features/feature_store.parquet')
    labels=pd.read_parquet(output/'labels.parquet')
    assert labels.future_revenue_12m.isna().all() and labels.future_revenue_28d.isna().all()
    assert labels.future_purchase_days_52w.notna().sum()==1
    for task,count in [('retention',2),('uplift',2),('clv',1)]:
        keys,x,y=load_campaign_inputs(output,task)
        assert len(keys)==len(x)==len(y)==count and x.shape[1]==39
        assert not {'treatment','canonical_type','churn','future_purchase_days_52w'}&set(x)
    assert features.isna().sum(axis=1).min()>20
    import json,sqlite3
    manifest=json.loads((output/'manifest.json').read_text())
    assert isinstance(manifest['model_sha256'],str) and manifest['model_asset_sha256']==DailyFixture.model_shas
    assert manifest['engineering_fixture_only']=='unit fixture' and manifest['production_export'] is False
    with sqlite3.connect(output/'raw.sqlite') as connection:
        assert pd.read_sql_query('SELECT * FROM coupon_redemptions',connection).empty
    with pytest.raises(FileExistsError):export_campaign(DailyFixture(),output,domain='tianchi',customers=2)


def test_daily_export_detects_an_incorrect_online_state(tmp_path):
    env=DailyFixture();env.corrupt_online=True
    with pytest.raises(AssertionError):export_campaign(env,tmp_path/'bad',domain='tianchi',customers=2)
    assert (tmp_path/'bad/failure.json').exists()


@pytest.mark.parametrize('mode,customers',[('auto',2),('none',3)])
def test_existing_environment_without_seed_identity_keeps_a_valid_assignment_audit(tmp_path,mode,customers):
    import json
    from simulate.v4.simulator.coupon_assignment import validate_coupon_assignments
    env=DailyFixture();env.customer_id_for_seed=None
    output=tmp_path/'legacy_env'
    export_campaign(env,output,domain='tianchi',customers=customers,coupon_assignment={'mode':mode})
    validate_coupon_assignments(pd.read_parquet(output/'treatments.parquet'),json.loads((output/'manifest.json').read_text()))
    with pytest.raises(ValueError,match='customer_id_for_seed'):
        export_campaign(env,tmp_path/'assigned',domain='tianchi',customers=customers,
            coupon_assignment={'mode':'assign','count':1})
    assert not (tmp_path/'assigned').exists()


@pytest.mark.parametrize('spec,treated',[
    ({'mode':'none'},0),
    ({'mode':'assign','ratio':0},0),
    ({'mode':'assign','ratio':1},3),
    ({'mode':'assign','count':1,'seed':19},1),
])
def test_campaign_modes_keep_factual_assignment_and_control_target_boundaries(tmp_path,spec,treated):
    import json
    from simulate.v4.simulator.coupon_assignment import validate_coupon_assignments
    output=tmp_path/'mode'
    report=export_campaign(DailyFixture(),output,domain='tianchi',customers=3,seed=42,coupon_assignment=spec)
    metadata=json.loads((output/'manifest.json').read_text())
    assignments=pd.read_parquet(output/'treatments.parquet')
    coupons=pd.read_parquet(output/'coupons.parquet')
    labels=pd.read_parquet(output/'labels.parquet')
    assert report['assignment_counts']=={'0':3-treated,'1':treated}
    assert len(coupons)==treated and labels.future_purchase_days_52w.notna().sum()==3-treated
    assert labels.future_revenue_12m.isna().all()
    assert labels.policy_id.eq('no_new_coupon').sum()==3-treated
    assert '-'+spec['mode']+'-' in metadata['dataset_id']
    validate_coupon_assignments(assignments,metadata)
    for task in ('retention','uplift'):
        keys,features,_=load_campaign_inputs(output,task)
        assert len(keys)==3 and features.shape[1]==39
    if treated==3:
        with pytest.raises(ValueError,match='CLV requires untreated'):
            load_campaign_inputs(output,'clv')
    else:assert len(load_campaign_inputs(output,'clv')[0])==3-treated


def test_campaign_explicit_ids_and_assignment_fail_before_output_creation(tmp_path):
    from simulate.v4.data.campaign_export import _seed
    chosen=str(_seed(42,'tianchi',1))
    output=tmp_path/'ids'
    export_campaign(DailyFixture(),output,domain='tianchi',customers=3,seed=42,
        coupon_assignment={'mode':'assign','customer_ids':[chosen]})
    coupons=pd.read_parquet(output/'coupons.parquet')
    assert coupons.customer_id.tolist()==[chosen]
    with pytest.raises(ValueError,match='unknown'):
        export_campaign(DailyFixture(),tmp_path/'bad-id',domain='tianchi',customers=3,seed=42,
            coupon_assignment={'mode':'assign','customer_ids':['unknown']})
    assert not (tmp_path/'bad-id').exists()


def test_daily_export_does_not_read_pending_or_future_days(tmp_path):
    env=DailyFixture();env.reset(5)
    from simulate.v4.data.campaign_export import _daily_snapshots
    first=_daily_snapshots(env.days,[{'customer_id':env.uid,'as_of_at':env.t0}])
    changed=env.days.copy()
    changed.loc[changed.known_at.gt(env.t0),['view','purchase']]=True
    pd.testing.assert_frame_equal(first,_daily_snapshots(changed,[{'customer_id':env.uid,'as_of_at':env.t0}]))
    with pytest.raises(ValueError,match='domain differs'):
        export_campaign(env,tmp_path/'wrong',domain='cosmetics',customers=2)


def test_combined_retail_ids_follow_calendar_and_preserve_order_foreign_keys():
    from simulate.v4.data.campaign_export import _global_event_ids
    at=pd.Timestamp('2024-01-01T00:00:00Z')
    tables={'events':pd.DataFrame({'event_id':[1,2,3,4],
        'event_at':[at+pd.Timedelta(days=i) for i in (2,3,0,1)],
        'event_type':['purchase','coupon_usage','purchase','coupon_usage'],
        'related_purchase_event_id':[None,1,None,3]}),
        'orders':pd.DataFrame({'order_id':['late','early'],'purchase_event_id':[1,3]})}
    _global_event_ids(tables)
    assert tables['events'].event_id.tolist()==[1,2,3,4]
    assert tables['events'].event_at.is_monotonic_increasing
    assert tables['orders'].purchase_event_id.tolist()==[3,1]
    assert tables['events'].related_purchase_event_id.dropna().tolist()==[1,3]


def test_campaign_calendar_preserves_each_later_observed_entry(tmp_path):
    class DifferentEntry(DailyFixture):
        def reset(self,seed):
            super().reset(seed);self.origin_clock=self.origin
            self.origin+=pd.Timedelta(days=seed%40)
            self.days=self.days.loc[self.days.day_start.ge(self.origin)].copy()
            self.days.loc[self.days.index[0],'view']=True
            history=pd.DataFrame({'user_id':self.uid,'day_start':self.days.day_start.map(model_date),
                'view':self.days.view,'purchase':self.days.purchase})
            values=public_observation(history,model_date(self.t0),source_contract={'feature_windows_days':[28,56,91]})
            return np.asarray([np.nan if values[k] is None else values[k] for k in EXPANDED_FEATURES])
    import json
    export_campaign(DifferentEntry(),tmp_path/'entries',domain='tianchi',customers=2,seed=42)
    metadata=json.loads((tmp_path/'entries/manifest.json').read_text())
    origin=weekly_calendar('2024-01-01T00:00:00+09:00')[0]
    assert pd.Timestamp(metadata['observation_start_at'])==origin
    assert all(pd.Timestamp(value)>=origin for value in metadata['customer_observation_start_at'].values())
    assert len(set(metadata['customer_observation_start_at'].values()))==2
