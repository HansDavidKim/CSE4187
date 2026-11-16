"""Joint source likelihood, availability clock and persistent-effect boundaries."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from simulate.v4.irl import fit_joint_latent_shopping as joint
from test_v4_latent_shopping import mdp


def test_joint_q_gradient_matches_full_customer_marginal():
    p, mask = mdp()
    counts = sparse.csr_matrix([[3, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2],
                               [0, 3, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1]])
    activity = pd.DataFrame(dict(user_id=['a','b','a'], label=[1,0,0], base_logit=[-.4,.3,.7]))
    incidence = sparse.csr_matrix([[1,0,1], [0,1,0]])
    k = 4
    base = np.arange(12).reshape(3,4)*.01
    offsets = joint.shopping.centered_offsets(np.arange(12).reshape(3,4)*.02-.1)
    returns = np.array([-.7,-.2,.3,.8])
    weights = np.array([.1,.2,.3,.4])
    _, _, q, _ = joint.joint_marginal(base, offsets, returns, weights, counts, incidence, activity, p, mask, .01)
    component_counts = np.asarray([(counts.T@q[:,z]).reshape(3,5) for z in range(k)])
    activity_q = q[[0,1,0]]
    flat = np.r_[base.ravel(),offsets[:-1].ravel(),returns]
    _, gradient = joint.joint_q_objective(flat,component_counts,activity,activity_q,p,mask,counts.sum()+len(activity),.01)
    def loss(value):
        return -joint.joint_marginal(*joint.unpack_parameters(value,3,k),weights,counts,incidence,activity,p,mask,.01)[0]
    for i in range(len(flat)):
        step=np.zeros_like(flat);step[i]=1e-5
        assert gradient[i] == pytest.approx((loss(flat+step)-loss(flat-step))/2e-5,abs=2e-6)


def test_activity_panel_has_completed_nonoverlap_windows_and_strict_context(monkeypatch):
    observations=[]
    def logit(model,context):
        observations.append(context.copy())
        assert context['has_observed_session']
        return float(context['session_count_30d'])
    monkeypatch.setattr(joint,'baseline_logit',logit)
    rows=pd.DataFrame([dict(user_id=uid, session=f'{uid}-{i}', at=pd.Timestamp(at),
        event_type='view',price_minor=100) for uid,times in [
            ('a',['2019-10-01T00:00Z','2019-10-31T00:00Z','2019-12-15T00:00Z']),
            ('b',['2019-11-15T00:00Z','2020-02-29T00:00Z'])] for i,at in enumerate(times)])
    frame,audit=joint.build_activity_panel(rows,{})
    assert (frame.target_at-frame['at']).eq(pd.Timedelta(days=30)).all()
    assert frame.loc[frame.user_id.eq('a'),'at'].diff().dropna().eq(pd.Timedelta(days=30)).all()
    assert frame.loc[frame.user_id.eq('b'),'at'].min()>pd.Timestamp('2019-11-15T00:00Z')
    first=frame.loc[frame.user_id.eq('a')].iloc[0]
    assert first.label==1 and first.base_logit==1  # at-boundary visit is label, never prior context
    assert frame.label.eq(0).any() and not frame.target_at.gt(pd.Timestamp(audit['observation_end'])).any()
    future=rows.copy();future.loc[future['at'].eq(pd.Timestamp('2020-02-29T00:00Z')),'price_minor']=9999999
    changed,_=joint.build_activity_panel(future,{})
    pd.testing.assert_frame_equal(frame,changed)


def test_activity_training_purges_end_at_cutoff_and_incomplete_boundaries():
    cutoff=pd.Timestamp(joint.CUTOFF)
    frame=pd.DataFrame(dict(at=[cutoff-pd.Timedelta(days=31),cutoff-pd.Timedelta(days=30),cutoff],
        target_at=[cutoff-pd.Timedelta(days=1),cutoff,cutoff+pd.Timedelta(days=30)],train_user=[True,True,False]))
    masks=joint.activity_masks(frame,cutoff)
    assert masks['train'].tolist()==[True,False,False]
    assert masks['user_and_time_holdout'].tolist()==[False,False,True]
    assert not any(mask.iloc[1] for mask in masks.values())


def _clock_case(monkeypatch):
    at=pd.Timestamp('2020-01-01T00:00Z')
    records=pd.DataFrame([
        dict(user_id='u',session=1,action=4,at=at,target_at=at+pd.Timedelta(minutes=30)),
        dict(user_id='u',session=2,action=0,at=at+pd.Timedelta(minutes=30),target_at=at+pd.Timedelta(minutes=30)),
        dict(user_id='u',session=3,action=0,at=at+pd.Timedelta(minutes=31),target_at=at+pd.Timedelta(minutes=31))])
    sessions=pd.DataFrame(dict(session=[1,2,3],invalid_total_cart=False,ambiguous_clock_sessions=False,
        is_censored=False,record_start=[0,1,2],record_end=[1,2,3]))
    data=dict(records=records,sessions=sessions,features=np.zeros((3,1)))
    monkeypatch.setattr(joint.shopping,'encode',lambda encoder,features:(np.zeros(len(features),dtype=int),np.zeros(len(features),dtype=bool)))
    policies=np.zeros((2,1,5));policies[0,0,[0,4]]=[.9,.1];policies[1,0,[0,4]]=[.1,.9]
    activity=pd.DataFrame([dict(user_id='u',at=at,target_at=at+pd.Timedelta(minutes=30),label=1,base_logit=0.)])
    sm={name:sessions.session.eq(sid) for name,sid in [('equal',2),('after',3)]}
    am={name:pd.Series(False,index=activity.index) for name in sm}
    return data,policies,activity,sm,am


def test_joint_prequential_excludes_same_clock_end_activity_and_future(monkeypatch):
    data,policies,activity,sm,am=_clock_case(monkeypatch)
    offsets=np.array([np.log(4),-np.log(4)])
    result=joint.prequential_metrics(data,{},policies,np.array([.5,.5]),activity,offsets,sm,am)
    assert result['equal']['action']['predicted_action_share']['view']==pytest.approx(.5)
    assert result['equal']['action']['median_available_action_activity_evidence_before_first_record']==[0,0]
    assert result['after']['action']['predicted_action_share']['view']==pytest.approx(.74)
    assert result['after']['action']['median_available_action_activity_evidence_before_first_record']==[2,1]
    delayed=activity.copy();delayed['target_at']+=pd.Timedelta(hours=1)
    future=joint.prequential_metrics(data,{},policies,np.array([.5,.5]),delayed,offsets,sm,am)
    assert future['equal']==result['equal']
    assert future['after']['action']['predicted_action_share']['view']==pytest.approx(.5)
    flip=activity.copy();flip['label']=0
    changed=joint.prequential_metrics(data,{},policies,np.array([.5,.5]),flip,offsets,sm,am)
    assert changed['equal']==result['equal']


def test_joint_activity_artifact_rejects_wrong_base_offsets_and_clock(monkeypatch):
    monkeypatch.setattr(joint.shopping,'validate_model',lambda model:None)
    head=dict(schema_version=1,type=joint.ACTIVITY_TYPE,horizon_days=30,base_model_sha256='a'*64,
        component_logit_offsets=[-.5,0.,.5,1.],cutoff_at=pd.Timestamp(joint.CUTOFF).isoformat(),
        fit_label_available_rule='window_end < cutoff',posterior_available_rule='window_end < decision_clock')
    block=dict(mixture_weights=[.25]*4,return_activity=head)
    joint.validate_model(block,'a'*64)
    assert joint.return_logit_offset(block,2)==.5
    assert joint.return_logit_offset({'mixture_weights':[.25]*4},2)==0
    for key,value in [('base_model_sha256','b'*64),('component_logit_offsets',[0.,1.]),
                      ('component_logit_offsets',[0.,1.,2.,float('nan')]),('horizon_days',28),
                      ('cutoff_at','2020-02-01T00:00:00+00:00'),('fit_label_available_rule','window_end <= cutoff')]:
        invalid=deepcopy(block);invalid['return_activity'][key]=value
        with pytest.raises(ValueError):joint.validate_model(invalid,'a'*64)
    for component in (True,-1,4,1.5):
        with pytest.raises(ValueError):joint.return_logit_offset(block,component)


def test_actual_small_joint_gem_is_monotone_with_control_and_activity_only_users():
    p,mask=mdp();records=[];states=[]
    for u in range(16):
        a=u%2
        for _ in range(1+u//4):
            records.append(dict(user_id=str(u),action=a));states.append(0)
            if a==0:
                second=3 if u%4==0 else 4
                records.append(dict(user_id=str(u),action=second));states.append(1)
                if second==4:
                    continue
            records.append(dict(user_id=str(u),action=4));states.append(2)
    panel={'records':pd.DataFrame(records)}
    activity=pd.DataFrame([dict(user_id=str(u),base_logit=-.3,label=int(u%4<2)) for u in range(20)])
    control=joint.fit_control(panel,p,mask,np.array(states),activity)
    model=joint.fit_joint_mixture(panel,p,mask,np.array(states),activity,control,max_em=2,max_m=30)
    assert len(model['users'])==20  # no-action customers' mature activity is not silently discarded
    assert model['fit']['normalizer']==len(records)+len(activity)
    assert model['fit']['maximum_bellman_residual']<1e-8
    for trace in model['fit']['trace']:
        assert trace['negative_Q_after']<=trace['negative_Q_before']+1e-8
        assert trace['regularized_complete_Q_after']>=trace['regularized_complete_Q_before']-1e-8
        assert trace['penalized_marginal_after']>=trace['penalized_marginal_before']-1e-8
    np.testing.assert_allclose(model['responsibility'].sum(axis=1),1)
    np.testing.assert_allclose(model['offsets'].sum(axis=0),0,atol=1e-12)
