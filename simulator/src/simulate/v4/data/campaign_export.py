"""Factual104-week campaign exports; daily observations keep their own grain."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd

from ..features import EXPANDED_FEATURES
from ..irl.source_contract import file_sha
from ..simulator.engine import SimulationResult
from ..simulator.raw_store import RAW_COLUMNS, load_raw_dataset
from ..storage import _sql_frame, materialize_features
from ..time_spec import weekly_metadata
from ..simulator.coupon_assignment import resolve_coupon_assignment, assign_coupons


def _json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def _seed(seed,domain,user):
    return int.from_bytes(hashlib.sha256(f'campaign-export:{seed}:{domain}:{user}'.encode()).digest()[:8],'big')


def _daily_weekly(days,uid,t0):
    rows=[]
    for lower in pd.date_range(t0-pd.Timedelta(days=91),t0,freq='7D',inclusive='left'):
        selected=days.loc[days.day_start.ge(lower)&days.day_start.lt(lower+pd.Timedelta(days=7))]
        rows.append(dict(customer_id=uid,as_of_at=t0,snapshot_id=f'{uid}:{t0.isoformat()}',reference_week=lower,
            sessions=int((selected.view|selected.purchase).sum()),purchases=int(selected.purchase.sum()),
            net_revenue_minor=np.nan,views=int(selected.view.sum()),session_seconds=np.nan))
    return rows


def _daily_sqlite(path,frames,metadata):
    with sqlite3.connect(path) as connection:
        for name,frame in frames.items():
            _sql_frame(frame).to_sql(name,connection,index=False,if_exists='fail')
        pd.DataFrame({'key':list(metadata),'value_json':[json.dumps(v,allow_nan=False) for v in metadata.values()]}).to_sql(
            'metadata',connection,index=False,if_exists='fail')
        if connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('daily raw SQLite corruption')


def _daily_snapshots(days, snapshots):
    """Rebuild the online39 from persisted, already released business dates."""
    from ..irl.tianchi_daily import public_observation
    from ..simulator.tianchi_campaign import model_date

    days=days.copy()
    for name in ('day_start','known_at'):
        days[name]=pd.to_datetime(days[name],utc=True)
    rows=[]
    for snapshot in snapshots:
        uid,at=snapshot['customer_id'],snapshot['as_of_at']
        past=days.loc[days.customer_id.eq(uid)&days.known_at.le(at)].copy()
        if not past.known_at.eq(past.day_start+pd.Timedelta(days=1)).all():
            raise ValueError('daily outcomes require next-business-day availability')
        history=pd.DataFrame({'user_id':past.customer_id.astype(str),
            'day_start':past.day_start.map(model_date),'view':past.view.astype(bool),
            'purchase':past.purchase.astype(bool)})
        values=public_observation(history,model_date(at),source_contract={'feature_windows_days':[28,56,91]})
        rows.append({'customer_id':uid,'as_of_at':at,'snapshot_id':f'{uid}:{at.isoformat()}',**values})
    result=pd.DataFrame(rows)
    result[list(EXPANDED_FEATURES)]=result[list(EXPANDED_FEATURES)].astype(float)
    return result


def _global_event_ids(tables):
    # Customer episodes overlap in calendar time. The raw-store ID contract
    # orders the combined event stream, with stable episode order for ties.
    events=tables['events'].sort_values('event_at',kind='stable').reset_index(drop=True)
    lookup=dict(zip(events.event_id,range(1,len(events)+1)))
    tables['events']=events
    for name in ('events','orders'):
        for column in ('event_id','related_purchase_event_id','purchase_event_id'):
            if column in tables[name]:
                tables[name][column]=tables[name][column].map(lambda v:lookup[v] if pd.notna(v) else None)


def export_campaign(env,output,*,domain,customers=128,seed=20261127,coupon_assignment=None):
    """Apply one factual t0 coupon assignment and retain every customer.

    No counterfactual outcome is presented as observed. Tianchi's control-only
    annual target is purchase-presence dates; its monetary CLV stays unavailable.
    """
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    spec=resolve_coupon_assignment(getattr(env,'settings',{}).get('coupon_assignment')
        if coupon_assignment is None else coupon_assignment)
    if type(customers) is not int or customers<1 or (spec['mode']=='auto' and (customers<2 or customers%2)):
        raise ValueError('auto requires an even factual cohort; other modes require at least one customer')
    if domain not in ('cosmetics','electronics','tianchi') or env.canonical_adapter._model.get('domain')!=domain:
        raise ValueError('export domain differs from the canonical campaign')
    if env.warmup_days!=364 or env.horizon_days!=364:raise ValueError('exports require52past+52future weeks')
    response=env.canonical_adapter._model
    if response.get('new_world_calibrated') is not True:raise ValueError('fresh-world coupon calibration is required')
    legacy=None
    if spec['mode']=='auto':
        legacy=np.r_[np.zeros(customers//2,dtype=int),np.ones(customers//2,dtype=int)]
        np.random.default_rng(seed).shuffle(legacy)
    identity=getattr(env,'customer_id_for_seed',None)
    if identity is None and spec['mode']=='assign':
        raise ValueError('assigned campaign exports require customer_id_for_seed on the environment')
    cohort=[str(identity(_seed(seed,domain,user))) if identity else str(user) for user in range(customers)]
    treatment,propensities,assignment_metadata=assign_coupons(cohort,spec,seed=seed,legacy=legacy)
    started=time.monotonic();output.mkdir(parents=True)
    frames={};snapshots=[];labels=[];assignments=[];daily_weeks=[];event_offset=0;customer_timing=[];customer_start={}
    model_shas=deepcopy(env.model_shas);daily=domain=='tianchi';metadata=None
    try:
        for user,(action,propensity) in enumerate(zip(treatment,propensities)):
            reset_started=time.monotonic()
            values=np.asarray(env.reset(_seed(seed,domain,user)),float)
            if values.shape!=(39,) or np.isinf(values).any():raise ValueError('campaign reset requires exactly39 finite-or-unavailable predictors')
            tenure=values[EXPANDED_FEATURES.index('observed_tenure_days')]
            recency=values[EXPANDED_FEATURES.index('visit_recency_days')]
            if np.isfinite(tenure) and tenure<91:raise ValueError('campaign exports require at least13weeks of observed history')
            if np.isfinite(recency) and recency>28:raise ValueError('campaign exports require activity in the preceding4weeks')
            reset_seconds=time.monotonic()-reset_started
            uid,t0=env.uid,env.now
            if identity is not None and str(uid)!=cohort[user]:raise ValueError('campaign customer ID differs from its declared seed identity')
            customer_start[uid]=env.origin.isoformat()
            snapshots.append({'customer_id':uid,'as_of_at':t0,'snapshot_id':f'{uid}:{t0.isoformat()}',**dict(zip(EXPANDED_FEATURES,values))})
            assignments.append(dict(customer_id=uid,as_of_at=t0,assigned_at=t0,treatment=int(action),
                assignment_probability=float(propensity),coupon_action=int(action),policy_id='no_new_coupon'))
            future_started=time.monotonic()
            for week in range(52):
                _,_,done,_=env.advance(int(action) if week==0 else 0,7)
                if bool(done)!=(week==51):raise ValueError('factual history lost a scheduled week')
            customer_timing.append({'customer_id':uid,'warmup_seconds':reset_seconds,
                'future52week_seconds':time.monotonic()-future_started,
                'initial_candidate_attempts':env.records['_metadata'].get('initial_candidate_attempts')})
            if env.model_shas!=model_shas:raise ValueError('campaign assets changed between customers')
            tables=env.tables;metadata=deepcopy(tables['_metadata']);end=env.now
            issued=tables.get('coupons',pd.DataFrame())
            if action:
                if len(issued)!=1:raise ValueError('assigned customers require exactly one t0 coupon')
                if not (pd.to_datetime(issued.sent_at,utc=True).eq(t0).all()
                        and pd.to_datetime(issued.expires_at,utc=True).eq(t0+pd.Timedelta(days=28)).all()):
                    raise ValueError('campaign coupons require t0 sending and exactly4weeks validity')
                if 'policy_id' not in issued:
                    issued=issued.copy();issued['policy_id']='coupon_once_t0_28d';tables['coupons']=issued
                policy_id=str(issued.policy_id.iloc[0])
            else:
                if len(issued):raise ValueError('untreated customers must not receive a coupon')
                policy_id='no_new_coupon'
            assignments[-1].update(policy_id=policy_id,actual_sent=bool(action),sent=bool(action),
                coupon_id=issued.coupon_id.iloc[0] if action else None,
                send_cost_minor=int(issued.send_cost_minor.sum()) if action else 0)
            follow=t0+pd.Timedelta(days=28)
            if daily:
                days=tables['raw_daily']
                future=days.loc[days.day_start.ge(t0)&days.day_start.lt(follow)]
                annual=days.loc[days.day_start.ge(t0)&days.day_start.lt(end)]
                retained=int((future.view|future.purchase).any());purchased=int(future.purchase.any())
                revenue28=revenue30=annual_revenue=np.nan
                annual_days=int(annual.purchase.sum()) if not action else np.nan
                daily_weeks.extend(_daily_weekly(days,uid,t0))
            else:
                events=tables['events'];sessions=tables['sessions'];orders=tables['orders']
                observed=set(events.loc[events.event_type.isin(['view','cart','remove_from_cart','purchase']),'session_id'])
                retained=int((sessions.session_id.isin(observed)&sessions.started_at.ge(t0)&sessions.started_at.lt(follow)).any())
                purchases=events.loc[events.event_type.eq('purchase'),['event_id','event_at']]
                paid=orders.merge(purchases,left_on='purchase_event_id',right_on='event_id',validate='one_to_one')
                purchased=int(paid.event_at.ge(t0).where(paid.event_at.lt(follow),False).any())
                def revenue(days):
                    return int(paid.loc[paid.event_at.ge(t0)&paid.event_at.lt(t0+pd.Timedelta(days=days)),'paid_minor'].sum())
                revenue28,revenue30=revenue(28),revenue(30)
                annual_revenue=revenue(364) if not action else np.nan;annual_days=np.nan
                # Per-episode IDs are local integers; global references must move together.
                lookup=dict(zip(events.event_id,range(event_offset+1,event_offset+len(events)+1)))
                for name in ('events','orders'):
                    for column in ('event_id','related_purchase_event_id','purchase_event_id'):
                        if column in tables[name]:
                            tables[name][column]=tables[name][column].map(lambda v:lookup[v] if pd.notna(v) else None)
                event_offset+=len(events)
            labels.append(dict(customer_id=uid,as_of_at=t0,snapshot_id=f'{uid}:{t0.isoformat()}',churn=1-retained,
                retained=retained,future_retained=retained,purchased_28d=purchased,
                future_revenue_28d=revenue28,future_revenue_30d=revenue30,future_revenue_12m=annual_revenue,
                future_purchase_days_52w=annual_days,outcome_end_at=follow,outcome_window_days=28,
                eligibility_window_days=28,definition_id='future_no_session_28d_active28d' if not daily else 'future_no_active_date_28d_active28d',
                is_censored=False,observation_end_at=end,clv_outcome_end_at=end,policy_id=policy_id))
            for name,frame in tables.items():
                if name=='_metadata':continue
                if name=='products' and name in frames:
                    pd.testing.assert_frame_equal(frame,frames[name][0]);continue
                frames.setdefault(name,[]).append(frame)
        if identity is None:
            _,_,assignment_metadata=assign_coupons([row['customer_id'] for row in assignments],spec,
                seed=seed,legacy=treatment)
        combined={name:pd.concat(parts,ignore_index=True) for name,parts in frames.items()}
        if not daily:_global_event_ids(combined)
        else:
            # No redemptions is an observed zero-row table, not an absent schema.
            for name,columns in {
                'coupons':['customer_id','coupon_id','sent_at','expires_at','send_cost_minor','discount_minor','policy_id'],
                'coupon_redemptions':['customer_id','coupon_id','purchase_day','known_at','discount_minor','paid_minor'],
            }.items():
                if name not in combined or not len(combined[name].columns):combined[name]=pd.DataFrame(columns=columns)
        combined.update(labels=pd.DataFrame(labels),treatments=pd.DataFrame(assignments))
        metadata.update(dataset_id=f'campaign104w-{domain}-{seed}',seed=seed,as_of_at=t0.isoformat(),
            observation_start_at=getattr(env,'origin_clock',env.origin).isoformat(),observation_end_at=end.isoformat(),
            outcome_window_days=28,eligibility_window_days=28,coupon_validity_days=28,min_customer_history_days=91,clv_label_days=364,
            definition_id='future_no_active_date_28d_active28d' if daily else 'future_no_session_28d_active28d',policy_id='randomized_once_t0',
            randomized_assignment=assignment_metadata['randomized'],assignment_design=assignment_metadata['design'],
            coupon_assignment=assignment_metadata,**weekly_metadata())
        if spec['mode']!='auto':metadata['dataset_id']+=f"-{spec['mode']}-{assignment_metadata['assignment_sha256'][:12]}"
        metadata['model_asset_sha256']=model_shas
        metadata['campaign_export_sha256']=file_sha(Path(__file__))
        metadata['coupon_assignment_runtime_sha256']=file_sha(Path(__file__).parents[1]/'simulator/coupon_assignment.py')
        metadata['customer_observation_start_at']=customer_start
        metadata['model_sha256']=hashlib.sha256(json.dumps(model_shas,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if response.get('engineering_fixture_only'):
            metadata['engineering_fixture_only']=response['engineering_fixture_only']
            metadata['production_export']=False
        expected=pd.DataFrame(snapshots)
        if daily:
            for name,frame in combined.items():frame.to_parquet(output/f'{name}.parquet',index=False)
            _daily_sqlite(output/'raw.sqlite',combined,metadata)
            feature_dir=output/'features';feature_dir.mkdir()
            with sqlite3.connect(output/'raw.sqlite') as connection:
                loaded=pd.read_sql_query('SELECT customer_id,day_start,known_at,view,purchase FROM raw_daily',connection)
            if len(loaded)!=len(combined['raw_daily']):raise ValueError('daily SQLite reload lost observations')
            _daily_snapshots(loaded,snapshots).to_parquet(feature_dir/'feature_store.parquet',index=False)
            combined['labels'].to_parquet(feature_dir/'labels.parquet',index=False)
            combined['treatments'].to_parquet(feature_dir/'treatments.parquet',index=False)
            pd.DataFrame(daily_weeks).to_parquet(feature_dir/'clv_weekly_inputs.parquet',index=False)
            _json(output/'manifest.json',metadata)
            feature_manifest={**weekly_metadata(),'predictors':list(EXPANDED_FEATURES),
                'data_contract':'tianchi_daily_campaign','clv_target':'future_purchase_days_52w',
                'clv_units':'purchase-presence dates; not monetary CLV','source_monetary_values_observed':False,
                'file_sha256':{p.name:file_sha(p) for p in feature_dir.iterdir()}}
            _json(feature_dir/'manifest.json',feature_manifest)
        else:
            # SimulationResult publishes atomically into a new directory.
            output.rmdir();SimulationResult(combined,time.monotonic()-started,metadata).save(output)
            loaded,_=load_raw_dataset(output/'raw.sqlite')
            if len(loaded['customers'])!=customers:raise ValueError('retail SQLite reload lost customers')
            materialize_features(output,output/'features')
            feature_dir=output/'features'
        actual=pd.read_parquet(feature_dir/'feature_store.parquet').set_index('customer_id')
        expected=expected.set_index('customer_id')
        np.testing.assert_allclose(actual.loc[expected.index,list(EXPANDED_FEATURES)],expected[list(EXPANDED_FEATURES)],equal_nan=True,rtol=1e-10,atol=1e-10)
        result={'status':'actual104week_factual_export_verified','domain':domain,'customers':customers,
            'production_export':not bool(response.get('engineering_fixture_only')),
            'assignment_counts':{'0':int((treatment==0).sum()),'1':int(treatment.sum())},
            'coupon_assignment':assignment_metadata,'weeks':[52,52],'predictors':list(EXPANDED_FEATURES),
            'public_predictor_count':39,'snapshot_raw_recalculation_matches':True,'sqlite_reload_verified':True,
            'control_only52week_clv':True,'clv_target':'future_purchase_days_52w' if daily else 'future_revenue_12m',
            'clv_target_units':'purchase-presence dates proxy' if daily else 'declared native minor units',
            'model_sha256':model_shas,'customer_runtime':customer_timing,
            'elapsed_seconds':time.monotonic()-started}
        _json(output/'verification.json',result)
        return result
    except BaseException as error:
        output.mkdir(parents=True,exist_ok=True)
        _json(output/'failure.json',{'type':type(error).__name__,'message':str(error),'customers_completed':len(labels),
            'requested_customers':customers,'customers_replaced_after_future':0,'customer_runtime':customer_timing})
        raise


def load_campaign_inputs(directory,task):
    """Read factual labels separately from exactly39 preassignment predictors."""
    directory=Path(directory);manifest=json.loads((directory/'features/manifest.json').read_text())
    if manifest.get('predictors')!=list(EXPANDED_FEATURES):raise ValueError('campaign predictor contract changed')
    for name,sha in manifest['file_sha256'].items():
        if file_sha(directory/'features'/name)!=sha:raise ValueError('campaign materialization changed')
    features=pd.read_parquet(directory/'features/feature_store.parquet')
    labels=pd.read_parquet(directory/'features/labels.parquet');treatment=pd.read_parquet(directory/'features/treatments.parquet')
    keys=['customer_id','as_of_at','snapshot_id']
    rows=features.merge(labels,on=keys,validate='one_to_one').merge(treatment[['customer_id','treatment']],on='customer_id',validate='one_to_one')
    if task=='retention':target='churn'
    elif task=='uplift':target='purchased_28d'
    elif task=='clv':
        target=manifest.get('clv_target','future_revenue_12m');rows=rows.loc[rows.treatment.eq(0)&rows[target].notna()]
    else:raise ValueError('task must be retention, uplift or clv')
    if rows.empty or rows[target].isna().any():
        raise ValueError('campaign task lacks mature factual targets; CLV requires untreated customers with52weeks of followup')
    return rows[keys+['treatment']],rows[list(EXPANDED_FEATURES)],rows[target]
