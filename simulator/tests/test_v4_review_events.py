import copy
import os
import gzip
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from simulate.v4.irl.fit_review_events import _fit_component, _features, _masks, load_amazon, load_olist, review_history_contexts
from simulate.v4.simulator.review_events import (ORDER_FEATURE_NAMES, REVIEW_FEATURE_NAMES, context_features,
    order_review_context, rating_probabilities, review_context, sample_rating, sample_review_schedule, validate_review_model)


def test_runtime_context_is_strictly_past_and_same_time_order_does_not_leak():
    t0=pd.Timestamp("2024-01-10",tz="UTC")
    history=[dict(event_at=t0-pd.Timedelta(days=2),rating=1),
             dict(event_at=t0-pd.Timedelta(days=1),rating=2),
             dict(event_at=t0-pd.Timedelta(days=1),rating=4)]
    expected=review_context(history,t0)
    assert expected==dict(prior_review_count=3,prior_mean_rating=7/3,last_rating=3.,days_since_review=1.)
    future=[dict(event_at=t0,rating=5),dict(event_at=t0+pd.Timedelta(days=1),rating=5)]
    assert review_context([*history,*future],t0)==expected
    orders=[t0-pd.Timedelta(days=7),t0,t0+pd.Timedelta(days=1)]
    assert order_review_context(orders,history,t0)=={**expected,"prior_order_count":1,"days_since_order":7.}
    assert review_context([],t0)["prior_review_count"]==0
    with pytest.raises(ValueError,match="timezone-aware"):
        review_context([],"2024-01-10")


def test_source_asof_features_equal_runtime_history_and_exclude_future_labels():
    source=pd.DataFrame(dict(user=["u","u","u","u","v"],
        at=pd.to_datetime(["2024-01-01","2024-01-02","2024-01-02","2024-01-03","2024-01-02"],utc=True),rating=[1,2,4,5,1]))
    framed=review_history_contexts(source,source)
    for index,row in framed.iterrows():
        local=source.loc[source.user.eq(row.user)]
        history=[dict(event_at=item.at,rating=item.rating) for item in local.itertuples(index=False)]
        np.testing.assert_allclose(_features(framed.loc[[index]],"product_stars")[0],context_features(review_context(history,row["at"])))
    changed=source.copy()
    changed.loc[changed["at"].ge(pd.Timestamp("2024-01-02",tz="UTC")),"rating"]=3
    cutoff=pd.Timestamp("2024-01-02",tz="UTC")
    original_features=_features(framed.loc[framed["at"].le(cutoff)],"product_stars")
    modified=review_history_contexts(changed,changed)
    np.testing.assert_array_equal(original_features,_features(modified.loc[modified["at"].le(cutoff)],"product_stars"))


def test_survey_training_label_windows_finish_before_time_holdout():
    frame=pd.DataFrame([dict(user=f"person{i}",at=pd.Timestamp("2020-01-01",tz="UTC")+pd.Timedelta(days=day))
                        for i in range(20) for day in range(120)])
    masks,cutoff=_masks(frame,outcome_days=28)
    assert (frame.loc[masks["train"],"at"]+pd.Timedelta(days=28)).le(cutoff).all()
    assert frame.loc[masks["time_holdout"],"at"].ge(cutoff).all()
    assert (~np.logical_or.reduce(list(masks.values()))).any()
    assert not set(frame.loc[masks["train"],"user"]) & set(frame.loc[masks["user_holdout"],"user"])


@pytest.fixture
def model():
    source=pd.DataFrame([dict(user=f"person{i}",at=pd.Timestamp("2020-01-01",tz="UTC")+pd.Timedelta(days=day),rating=1+i%5)
                         for i in range(80) for day in range(5)])
    frame=review_history_contexts(source,source)
    stars,_=_fit_component(frame,"product_stars")
    width=len(ORDER_FEATURE_NAMES)
    survey=dict(model=dict(feature_names=list(ORDER_FEATURE_NAMES),classes=[0,1],coef=[[0.]*width],intercept=[700.],
        scaler_mean=[0.]*width,scaler_scale=[1.]*width,feature_min=[-10.]*width,feature_max=[10.]*width),window_days=28,
        delay_score_banks=dict(pooled=[[60.,1],[120.,5]],first_observed_order=[[60.,1],[120.,5]],repeat_observed_order=[]))
    return dict(schema_version=1,type="source_conditional_review_events",product_stars=stars,order_survey=survey)


def test_learned_stars_use_prior_preferences_and_record_real_holdouts(model):
    validate_review_model(model)
    at=pd.Timestamp("2024-01-10",tz="UTC")
    low=review_context([dict(event_at=at-pd.Timedelta(days=1),rating=1)],at)
    high=review_context([dict(event_at=at-pd.Timedelta(days=1),rating=5)],at)
    p_low,p_high=rating_probabilities(model,low),rating_probabilities(model,high)
    assert np.isclose(sum(p_low),1) and p_low[0]>p_high[0] and p_high[4]>p_low[4]
    metrics=model["product_stars"]["validation"]
    assert all(metrics[name]["records"]>0 for name in ("train","user_holdout","time_holdout","user_and_time_holdout"))
    assert metrics["train"]["nll"]<metrics["train"]["frequency_baseline_nll"]
    assert sample_rating(model,low,np.random.default_rng(7))==sample_rating(model,low,np.random.default_rng(7))


def test_schedule_samples_observed_pairs_and_never_feeds_order_score_into_stars(model):
    at=pd.Timestamp("2024-01-10",tz="UTC")
    context=order_review_context([],[],at)
    result=sample_review_schedule(model,context,np.random.default_rng(7))
    assert result["scheduled"] and [result["delay_seconds"],result["source_order_score"]] in [[60.,1],[120.,5]]
    before=rating_probabilities(model,context)
    changed=copy.deepcopy(model)
    changed["order_survey"]["delay_score_banks"]["first_observed_order"]=[[360.,2]]
    assert rating_probabilities(changed,context)==before
    model["order_survey"]["model"]["intercept"]=[-700.]
    assert not sample_review_schedule(model,context,np.random.default_rng(7))["scheduled"]


@pytest.mark.parametrize("case",["shape","bounds","clock","classes"])
def test_invalid_artifact_is_rejected(model,case):
    if case=="shape": model["product_stars"]["model"]["coef"]=[]
    elif case=="bounds": model["product_stars"]["model"]["feature_min"]=[999.]*len(REVIEW_FEATURE_NAMES)
    elif case=="clock": model["order_survey"]["delay_score_banks"]["pooled"]=[[28*86400,5]]
    else: model["product_stars"]["model"]["classes"]=[1,2,3,4]
    with pytest.raises(ValueError): validate_review_model(model)


def test_amazon_reader_retains_only_valid_verified_metadata(tmp_path):
    path=tmp_path/"reviews.gz"
    rows=[dict(user_id="u",asin="i",rating=5.,timestamp=1600000000000,verified_purchase=True),
          dict(user_id="u",asin="j",rating=1.,timestamp=1600000000000,verified_purchase=False),
          dict(user_id="u",asin="k",rating=9.,timestamp=1600000000000,verified_purchase=True)]
    with gzip.open(path,"wt") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    frame,quality=load_amazon(path)
    assert frame.rating.tolist()==[5] and quality["excluded_not_verified_purchase"]==1 and quality["excluded_invalid_metadata"]==1


def test_olist_clocks_exclude_invalid_and_censored_orders_without_lifetime_zero_labels(tmp_path):
    people=pd.DataFrame(dict(customer_id=["a","b","c","d"],customer_unique_id=["u","u","v","v"]))
    orders=pd.DataFrame(dict(order_id=["o1","o2","o3","o4"],customer_id=people.customer_id,
        order_purchase_timestamp=["2024-01-01","2024-01-05","2024-01-10","2024-04-01"]))
    reviews=pd.DataFrame(dict(review_id=["r1","r2","r3","r4","r5"],order_id=["o1","o1","o2","o3","o4"],
        review_score=[5,1,2,1,4],review_creation_date=["2024-01-02"]*5,
        review_answer_timestamp=["2024-01-03","2024-01-04","2024-02-02","2024-01-09","2024-04-02"]))
    for name,frame in [("customers",people),("orders",orders),("order_reviews",reviews)]:
        frame.to_csv(tmp_path/f"olist_{name}_dataset.csv",index=False)
    frame,quality=load_olist(tmp_path)
    indexed=frame.set_index("order_id")
    assert set(indexed.index)=={"o1","o2"} and indexed.rating.to_dict()=={"o1":1,"o2":0}
    assert indexed.loc["o2","prior_order_count"]==1 and indexed.loc["o2","prior_review_count"]==1
    assert quality["answer_before_purchase_rows"]==1 and quality["responses_outside28d_not_lifetime_nonresponse"]==1
    assert quality["excluded_not_mature_28d_or_bad_purchase_clock"]==1


def test_cli_refuses_existing_output_before_reading_or_training(tmp_path):
    output=tmp_path/"existing.json"
    output.write_text("keep this artifact")
    process=subprocess.run([sys.executable,"-m","simulate.v4.irl.fit_review_events","--amazon",str(tmp_path/"missing.gz"),
                            "--output",str(output)],capture_output=True,text=True,env={**os.environ,"PYTHONPATH":os.pathsep.join(sys.path)})
    assert process.returncode!=0 and "artifact already exists" in process.stderr
    assert output.read_text()=="keep this artifact"


def test_star_window_selection_never_reads_outer_holdout_labels():
    source=pd.DataFrame([dict(user=f"person{i}",at=pd.Timestamp("2010-01-01",tz="UTC")+pd.Timedelta(days=30*month),rating=1+i%5)
                         for i in range(80) for month in range(72)])
    frame=review_history_contexts(source,source)
    before,masks=_fit_component(frame,"product_stars")
    heldout_users=set(frame.loc[masks["user_holdout"],"user"])
    cutoff=pd.Timestamp(before["validation"]["time_cutoff"])
    changed=source.copy()
    selected=changed.user.isin(heldout_users)|changed["at"].ge(cutoff)
    changed.loc[selected,"rating"]=6-changed.loc[selected,"rating"]
    after,_=_fit_component(review_history_contexts(changed,changed),"product_stars")
    assert after["model"]==before["model"]
    assert after["training_selection"]==before["training_selection"]
    selection=before["training_selection"]
    assert len(selection["candidates"])==8 and selection["inner_validation_records"]>0
    assert pd.Timestamp(selection["inner_time_cutoff"])<cutoff
    assert len({row["fit_records"] for row in selection["candidates"]})>1


def test_delay_banks_keep_every_training_pair_above_old_compression_limit(tmp_path,model,monkeypatch):
    from simulate.v4.irl import fit_review_events as fitting
    amazon=tmp_path/"amazon.gz"
    amazon.write_bytes(b"fixture source")
    for name in ("orders","customers","order_reviews"):
        (tmp_path/f"olist_{name}_dataset.csv").write_text("fixture source")
    raw=pd.DataFrame(dict(user=["u"],at=[pd.Timestamp("2020-01-01",tz="UTC")],rating=[5]))
    frame=pd.DataFrame(dict(prior_order_count=np.zeros(4104),rating=np.ones(4104),
                            delay_seconds=np.arange(1,4105),source_order_score=1+np.arange(4104)%5))
    mask=np.arange(len(frame))<4101
    monkeypatch.setattr(fitting,"load_amazon",lambda path:(raw,{}))
    monkeypatch.setattr(fitting,"load_olist",lambda path:(frame,{}))
    monkeypatch.setattr(fitting,"_fit_component",lambda rows,kind:(copy.deepcopy(model[kind]),{"train":mask}))
    fitted=fitting.fit_review_events(amazon,tmp_path)
    contract=fitted["runtime_event_contract"]
    assert contract["raw_schema_version"]==5 and contract["product_id"] is not None
    assert contract["order_line_no"] and contract["session_id"] is None
    expected=[[float(index),int(1+(index-1)%5)] for index in range(1,4102)]
    assert fitted["order_survey"]["delay_score_banks"]["pooled"]==expected
    assert fitted["order_survey"]["delay_score_banks"]["first_observed_order"]==expected
    assert fitted["order_survey"]["delay_score_banks"]["repeat_observed_order"]==[]
