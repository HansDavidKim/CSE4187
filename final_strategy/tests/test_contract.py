"""Small engineering fixtures verify contracts; they are not model-performance evidence."""
from dataclasses import asdict, replace
from decimal import Decimal
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from final_strategy import (DecisionConfig, PredictionBatch, PredictionMetadata, build_decisions,
                            integrate_inputs, load_feature_store, write_outputs)
from final_strategy.evaluation import _holm, qini_curves, strategy_evaluation
from final_strategy.pipeline import CUSTOMER_COLUMNS, SEGMENTS


@pytest.fixture
def fixture():
    at = pd.Timestamp("2025-01-06T00:00:00+09:00").tz_convert("UTC")
    features = pd.DataFrame({"dataset_id": "engineering-fixture", "customer_id": [f"u{x}" for x in range(8)],
        "as_of_at": at, "snapshot_id": [f"u{x}:t0" for x in range(8)], "reference_week": at,
        "currency": "SOURCE_UNSPECIFIED", "amount_unit": "0.01", "raw_database_sha256": "a"*64,
        "observed_tenure_days": [30, 100, 200, 300, 400, 500, 600, 700],
        "canonical_type": "must-not-be-used", "future_revenue_12m": 999999})
    identity = features.drop(columns=["observed_tenure_days", "canonical_type", "future_revenue_12m"]).copy()
    identity["reference_week"] = "2025-01-06"
    identity["input_cutoff_at"] = at

    def batch(role, values, version="fixture-model"):
        frame = identity.assign(prediction_end_at=at+pd.Timedelta(weeks=52 if role=="clv" else 4), **values)
        target, semantics = {"churn": ("future_no_session_28d_active28d", "probability"),
            "uplift": ("purchase_in_28d", "probability_pair"), "clv": ("clv_paid_amount_52w_v1", "nonnegative_amount"),
            "gross_4w": ("gross_amount_4w_if_purchase_v1", "nonnegative_amount")}[role]
        return PredictionBatch(frame, PredictionMetadata(version, target, 52 if role=="clv" else 4, "weeks", semantics))
    predictions = {"churn": batch("churn", {"churn_probability": [.8,.8,.8,.2,.9,.2,.7,.8]}),
        "uplift": batch("uplift", {"p0": [.1,.1,.2,.8,.2,.7,.6,.7], "p1": [.4,.5,.5,.81,.21,.71,.2,.5]}),
        "clv": batch("clv", {"predicted_clv_12m": [200,500,50,400,600,40,700,60]}),
        "gross_4w": batch("gross_4w", {"predicted_gross_4w_if_purchase": 1000.})}
    config = DecisionConfig(60,10,"SOURCE_UNSPECIFIED","0.01",100,.5,.05,91,
                            ("newly_observed","high_value_persuadable","low_value_persuadable","high_value_sure_thing"),.5,20)
    return features, predictions, config, batch


def test_exact10_columns_six_business_rules_costs_and_budget_what_if(fixture):
    features, predictions, config, _ = fixture
    features.index = np.arange(len(features))*2
    output = build_decisions(features, predictions, config)
    assert output.customers.columns.tolist() == CUSTOMER_COLUMNS
    assert set(output.customers.segment) == set(SEGMENTS)
    assert output.customers.send.sum() == 2
    assert set(output.customers.loc[output.customers.send.eq(1), "customer_id"]) == {"u0","u1"}
    assert output.customers.loc[1,"expected_net_value"] == pytest.approx(370.)
    assert output.budget_summary["expected_cost"] == 60
    assert output.budget_summary["expected_send_cost"] == 20
    assert output.budget_summary["expected_discount_cost"] == 40
    assert output.budget_summary["expected_incremental_revenue"] == pytest.approx(700)
    assert output.what_if.send_count.tolist() == [1,2,3]
    assert output.what_if.expected_cost.le(output.what_if.total_budget).all()
    assert output.customers.loc[[6,7],"send"].eq(0).all()
    assert output.segment_summary.customers.sum() == 8 and output.segment_summary.share.sum() == pytest.approx(1.)
    assert not {"canonical_type","future_revenue_12m","observed_tenure_days"} & set(output.customers)


def test_missing4w_gross_does_not_use52w_clv_as_money(fixture):
    features,predictions,config,_=fixture
    output=build_decisions(features,{k:v for k,v in predictions.items() if k!="gross_4w"},config)
    assert output.customers.clv.notna().all()
    assert output.customers.expected_net_value.isna().all() and output.customers.send.eq(0).all()
    assert output.budget_summary["expected_cost"] is None and output.budget_summary["expected_roi"] is None
    assert output.budget_summary["status"] == "unavailable"
    assert output.availability.reason.str.contains("4-week conditional gross").all()


def test_tianchi_unobserved_money_is_not_zero_profit(fixture):
    features,predictions,config,_=fixture
    features.attrs["feature_manifest"]={"source_monetary_values_observed":False}
    output=build_decisions(features,predictions,config)
    assert output.customers.expected_net_value.isna().all() and output.customers.send.eq(0).all()
    assert output.budget_summary["expected_incremental_revenue"] is None


def test_missing_core_prediction_batches_are_clear_errors(fixture):
    features,predictions,config,_=fixture
    with pytest.raises(ValueError,match="Missing required.*uplift.*clv"):
        build_decisions(features,{"churn":predictions["churn"]},config)


def test_mixed_policy_clv_bundle_is_not_silently_used_as_no_coupon_value(fixture):
    features,predictions,config,_=fixture
    predictions["clv"].producer_metadata={"policy_condition":"observed_experiment_mix"}
    with pytest.raises(ValueError,match="control_no_coupon"):
        build_decisions(features,predictions,config)


@pytest.mark.parametrize("role,column",[("churn","raw_database_sha256"),("uplift","snapshot_id"),("clv","currency"),("gross_4w","amount_unit")])
def test_prediction_source_identity_cannot_be_omitted_to_bypass_checks(fixture,role,column):
    features,predictions,config,_=fixture
    predictions[role].frame=predictions[role].frame.drop(columns=column)
    with pytest.raises(ValueError,match="required provenance"):
        build_decisions(features,predictions,config)


@pytest.mark.parametrize("fault",["cutoff","horizon","target","cohort","alias","delta","future","duplicate"])
def test_prediction_contract_rejects_leakage_and_semantic_mismatch(fixture,fault):
    features,predictions,config,_=fixture
    p=predictions["uplift"]
    if fault=="cutoff":p.frame["input_cutoff_at"]+=pd.Timedelta(days=1)
    if fault=="horizon":p.metadata=replace(p.metadata,horizon_value=52)
    if fault=="target":p.metadata=replace(p.metadata,target_definition_id="future_churn")
    if fault=="cohort":p.frame=p.frame.iloc[:-1]
    if fault=="alias":p.frame["purchase_probability_no_coupon"]=0.
    if fault=="delta":p.frame["uplift_score"]=0.
    if fault=="future":p.frame["prediction_end_at"]+=pd.Timedelta(days=1)
    if fault=="duplicate":p.frame=pd.concat([p.frame,p.frame.iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError):build_decisions(features,predictions,config)


def test_missing_tenure_is_not_invented_new_or_mature_status(fixture):
    features,predictions,config,_=fixture
    output=build_decisions(features.drop(columns="observed_tenure_days"),predictions,config)
    assert output.customers.segment.isna().all() and output.customers.send.eq(0).all()
    assert output.metadata["unavailable_segment_customers"]==8


def test_positive_purchase_uplift_is_not_called_sure_thing_due_to_low_churn(fixture):
    features,predictions,config,_=fixture
    predictions["uplift"].frame.loc[3,"p1"] = .95
    output=build_decisions(features,predictions,config)
    assert output.graph_data["uplift_quadrants"].loc[3,"response_group"] == "Persuadables"
    assert output.customers.loc[3,"segment"] == "high_value_no_action"
    assert output.customers.loc[3,"send"] == 0


def test_zero_cost_is_not_infinite_roi_and_unknown_discount_is_not_free(fixture):
    features,predictions,config,_=fixture
    free=build_decisions(features,predictions,replace(config,budget=0,send_cost=0,expected_discount_per_send=0))
    assert free.customers.send.sum() == 4
    assert free.budget_summary["expected_roi"] is None
    assert free.budget_summary["roi_reason"] == "zero expected cost"
    missing=build_decisions(features,predictions,replace(config,expected_discount_per_send=None))
    assert missing.customers.expected_net_value.isna().all() and missing.customers.send.eq(0).all()


def test_discount_estimates_require_same_cutoff_and_units(fixture):
    features,predictions,config,_=fixture
    costs=features[["dataset_id","customer_id","as_of_at","currency","amount_unit"]].assign(expected_discount_if_sent=20.)
    with pytest.raises(ValueError,match="time endpoints"):
        build_decisions(features,predictions,replace(config,expected_discount_per_send=None),expected_discounts=costs)
    costs["input_cutoff_at"]=costs.as_of_at
    costs["prediction_end_at"]=costs.as_of_at+pd.Timedelta(weeks=4)
    output=build_decisions(features,predictions,replace(config,expected_discount_per_send=None),expected_discounts=costs)
    assert output.budget_summary["expected_cost"]==60.


def factual(features):
    at=features.as_of_at.iloc[0]
    obs=features[["dataset_id","customer_id","as_of_at"]].assign(assigned_at=at,
        outcome_end_at=at+pd.Timedelta(weeks=4),observation_end_at=at+pd.Timedelta(weeks=52),
        treatment=[0,1,0,1,0,1,0,1],assignment_probability=.5,purchased_28d=[0,1,1,1,0,0,0,1])
    metadata={"randomized_assignment":True,"evaluation_only":True,"development_customer_keys":[{"dataset_id":"training","customer_id":"train0"}],
              "models_frozen_at":(at-pd.Timedelta(days=1)).isoformat()}
    return obs,metadata


def test_qini_multiple_models_and_random_targeting_reference(fixture):
    features,predictions,_,batch=fixture
    obs,metadata=factual(features)
    second=batch("uplift",{"p0":[.2]*8,"p1":[.3]*8},version="fixture-other")
    result=qini_curves(obs,{"fixture-first":predictions["uplift"],"fixture-other":second},metadata)
    assert result["status"]=="available"
    for _,rows in result["frame"].groupby("model"):
        assert rows.iloc[0].cumulative_gain==0
        np.testing.assert_allclose(rows.random_targeting_gain,rows.fraction*rows.iloc[-1].cumulative_gain)
    # Score blocks of sizes 1,2,3,1,1; their IPW gain increments are 1/4,-1/4,1/4,1/4,0.
    assert result["qini_coefficients"]["fixture-first"] == -3 / 64
    assert result["qini_coefficients"]["fixture-other"] == 0.
    constant=result["frame"].loc[result["frame"].model.eq("fixture-other")]
    assert len(constant)==2
    shuffled=qini_curves(obs.sample(frac=1,random_state=4),{"fixture-first":predictions["uplift"],"fixture-other":second},metadata)
    pd.testing.assert_frame_equal(result["frame"],shuffled["frame"])
    metadata["coupon_assignment"]={"randomized":False}
    assert qini_curves(obs,{"model":second},metadata)["status"]=="unavailable"


def test_qini_rejects_single_arm_and_model_development_overlap(fixture):
    features,predictions,_,_=fixture
    obs,metadata=factual(features)
    obs["treatment"]=0
    assert qini_curves(obs,{"m":predictions["uplift"]},metadata)["status"]=="unavailable"
    obs,metadata=factual(features)
    metadata["development_customer_keys"]=[{"dataset_id":"engineering-fixture","customer_id":"u0"}]
    assert "overlap" in qini_curves(obs,{"m":predictions["uplift"]},metadata)["reason"]


def strategy_factual():
    at=pd.Timestamp("2025-01-06T00:00:00+09:00").tz_convert("UTC")
    rows=[]
    for name,purchases,churns in [("A",8,12),("B",12,6),("C",16,2)]:
        for i in range(20):
            purchased=int(i<purchases);sent=int(name!="A")
            rows.append(dict(dataset_id="strategy-fixture",customer_id=f"{name}{i}",as_of_at=at,assigned_at=at,
                outcome_end_at=at+pd.Timedelta(weeks=4),observation_end_at=at+pd.Timedelta(weeks=52),strategy_group=name,
                snapshot_id=f"{name}{i}:t0",raw_database_sha256="a"*64,assignment_probability=1/3,
                outcome_observed=True,is_censored=False,actual_sent=bool(sent),
                purchased_28d=purchased,churn=int(i>=20-churns),gross_revenue_minor=1000*purchased,
                discount_minor=100*purchased*sent,send_cost_minor=100*sent,currency="SOURCE_UNSPECIFIED",amount_unit="0.01"))
    _,meta=factual(pd.DataFrame({"dataset_id":["x"]*8,"customer_id":list("abcdefgh"),"as_of_at":at}))
    meta["strategy_groups"]=["A","B","C"]
    meta["strategy_randomized_assignment"]=True
    meta["primary_metric"]="churn_rate"
    return pd.DataFrame(rows),meta


def assigned_roster(obs):
    return obs[["dataset_id", "customer_id", "as_of_at", "snapshot_id", "raw_database_sha256",
                "assigned_at", "strategy_group", "assignment_probability"]].copy()


def test_strategy_factual_groups_ci_holm_costs_and_baseline():
    obs,metadata=strategy_factual()
    result=strategy_evaluation(obs,metadata,assignments=assigned_roster(obs))
    assert result["status"]=="available" and result["baseline"]=="A"
    frame=result["frame"].set_index("strategy_group")
    assert frame.customers.tolist()==[20,20,20]
    assert frame.loc["B","cost"]==3200
    assert frame.loc["B","net_revenue_per_customer"]==440
    assert frame.loc["B","roi"]==pytest.approx(.25)
    assert frame.loc["B","difference_vs_A"]==pytest.approx(-.3)
    for group in ["B","C"]:
        assert frame.loc[group,"adjusted_p_value"]>=frame.loc[group,"p_value"]
        assert frame.loc[group,"difference_ci95_lower"]<frame.loc[group,"difference_vs_A"]<frame.loc[group,"difference_ci95_upper"]
    assert frame.loc["A","difference_vs_A"]==0 and pd.isna(frame.loc["A","p_value"])
    np.testing.assert_allclose(_holm([.01,.04,.03]),[.03,.06,.06])


def test_paid_and_net_revenue_are_not_discounted_twice_or_aliased_as_gross():
    obs,metadata=strategy_factual()
    obs=obs.loc[obs.strategy_group.isin(["A","B"])].copy()
    sent=obs.strategy_group.eq("B")
    obs["purchased_28d"]=1
    obs["gross_revenue_minor"]=np.where(sent,100.,80.)
    obs["discount_minor"]=np.where(sent,10.,0.)
    obs["send_cost_minor"]=sent.astype(float)
    obs["paid_revenue_minor"]=obs.gross_revenue_minor-obs.discount_minor
    obs["net_revenue_minor"]=obs.paid_revenue_minor
    obs["net_after_marketing_minor"]=obs.net_revenue_minor-obs.send_cost_minor
    metadata.update(strategy_groups=["A","B"],primary_metric="net_revenue_per_customer")
    obs["assignment_probability"] = .5
    result=strategy_evaluation(obs,metadata,primary_metric="net_revenue_per_customer",assignments=assigned_roster(obs))["frame"].set_index("strategy_group")
    assert result.loc["A","net_revenue_per_customer"]==80
    assert result.loc["B","net_revenue_per_customer"]==89
    assert obs.groupby("strategy_group").gross_revenue_minor.mean().diff().loc["B"]==20
    assert result.loc["B","cost"]/result.loc["B","customers"]==11
    assert result.loc["B","difference_vs_A"]==9
    assert result.loc["B","roi"]==pytest.approx(9/11)
    assert obs.loc[sent,"net_revenue_minor"].eq(90).all()
    assert obs.loc[sent,"net_after_marketing_minor"].eq(89).all()
    for field in ("paid_revenue_minor","net_revenue_minor","net_after_marketing_minor"):
        wrong=obs.copy();wrong.loc[wrong.strategy_group.eq("B"),field]-=10
        with pytest.raises(ValueError,match="cannot substitute for gross"):
            strategy_evaluation(wrong,metadata)
    paid_only=strategy_evaluation(obs.drop(columns="gross_revenue_minor"),metadata)["frame"]
    assert paid_only.roi.isna().all() and paid_only.net_revenue_per_customer.isna().all()
    no_purchase=obs.copy()
    index=no_purchase.index[no_purchase.strategy_group.eq("B")][0]
    no_purchase.loc[index,["purchased_28d","gross_revenue_minor","discount_minor","paid_revenue_minor","net_revenue_minor","net_after_marketing_minor"]]=[0,0,0,0,0,-1]
    checked=strategy_evaluation(no_purchase,metadata)["frame"].set_index("strategy_group")
    assert checked.loc["B","net_revenue_per_customer"]==pytest.approx((19*89-1)/20)


def test_strategy_requires_original_itt_and_complete_followup():
    obs, meta = strategy_factual()
    original = assigned_roster(obs)
    # B0 is a failed send; A0 was never targeted. Both remain in the original population.
    obs.loc[obs.customer_id.eq("B0"), ["actual_sent", "send_cost_minor"]] = [False, 0]
    assert strategy_evaluation(obs, meta, assignments=original)["status"] == "available"
    for absent in (["B0"], ["A0"], [f"B{i}" for i in range(10)]):
        with pytest.raises(ValueError, match="original assignment"):
            strategy_evaluation(obs.loc[~obs.customer_id.isin(absent)], meta, assignments=original)
    for column, value in (("strategy_group", "A"), ("assignment_probability", .9)):
        changed = obs.copy()
        changed.loc[changed.customer_id.eq("B0"), column] = value
        with pytest.raises(ValueError, match="original assignment"):
            strategy_evaluation(changed, meta, assignments=original)
    with pytest.raises(ValueError, match="Duplicate"):
        strategy_evaluation(obs, meta, assignments=pd.concat([original, original.iloc[:1]]))
    with pytest.raises(ValueError, match="probability"):
        strategy_evaluation(obs, meta, assignments=original.assign(assignment_probability=1))
    for rows, roster in ((obs, None), (obs.assign(is_censored=True, outcome_observed=False), original),
                         (obs.assign(outcome_observed=False), original), (obs.assign(is_censored=True), original),
                         (obs.drop(columns="outcome_observed"), original)):
        result = strategy_evaluation(rows, meta, assignments=roster)
        assert result["status"] == "descriptive_only"
        assert result["frame"][["roi", "p_value", "difference_ci95_lower"]].isna().all().all()
    with pytest.raises(ValueError, match="boolean"):
        strategy_evaluation(obs.assign(outcome_observed="True"), meta, assignments=original)


@pytest.mark.parametrize("invalid", [Decimal(".5"), True, np.inf, np.nan, -1])
def test_realized_minor_rejects_invalid_charge(invalid):
    obs, meta = strategy_factual()
    obs["send_cost_minor"] = obs.send_cost_minor.astype(object)
    obs.loc[0, "send_cost_minor"] = invalid
    with pytest.raises(ValueError, match="minor"):
        strategy_evaluation(obs, meta, assignments=assigned_roster(obs))


def test_accounting_preserves_minor_above_float_and_int64_limits():
    obs, meta = strategy_factual()
    obs["purchased_28d"] = 1
    obs["gross_revenue_minor"] = np.int64(2**62)
    obs["discount_minor"] = 0
    obs["paid_minor"] = [Decimal(2**62)] * len(obs)
    result = strategy_evaluation(obs, meta, assignments=assigned_roster(obs))
    assert result["status"] == "available"
    obs.loc[0, "paid_minor"] += 1
    with pytest.raises(ValueError, match="contradicts"):
        strategy_evaluation(obs, meta, assignments=assigned_roster(obs))
    obs["gross_revenue_minor"] = np.int64(2**53)
    obs["paid_minor"] = np.int64(2**53 + 1)
    with pytest.raises(ValueError, match="contradicts"):
        strategy_evaluation(obs, meta, assignments=assigned_roster(obs))
    obs = obs.drop(columns="paid_minor")
    obs["send_cost_minor"] = np.int64(2**62)
    obs["actual_sent"] = True
    table = strategy_evaluation(obs, meta, assignments=assigned_roster(obs))["frame"]
    assert table.cost.tolist() == [20 * 2**62] * 3 and table.roi.isna().all()
    obs.loc[obs.strategy_group.ne("A"), "send_cost_minor"] = 0
    assert strategy_evaluation(obs, meta, assignments=assigned_roster(obs))["frame"].roi.isna().all()
    with pytest.raises(ValueError, match="positive charge"):
        strategy_evaluation(obs.assign(actual_sent=False), meta, assignments=assigned_roster(obs))


def test_nonrandom_strategy_and_missing_money_do_not_get_invented_roi_or_p():
    obs,metadata=strategy_factual()
    metadata["randomized_assignment"]=False
    output=strategy_evaluation(obs,metadata)
    assert output["status"]=="descriptive_only"
    assert output["frame"].p_value.isna().all() and output["frame"].roi.isna().all()
    assert output["frame"].primary_ci95_lower.isna().all()
    metadata["randomized_assignment"]=True
    metadata["source_monetary_values_observed"]=False
    output=strategy_evaluation(obs,metadata)
    assert output["frame"].cost.isna().all() and output["frame"].roi.isna().all()


@pytest.mark.parametrize("missing",["strategy_groups","primary_metric","strategy_randomized_assignment"])
def test_undeclared_comparison_family_or_primary_metric_stays_descriptive(missing):
    obs,metadata=strategy_factual()
    metadata.pop(missing)
    output=strategy_evaluation(obs,metadata)
    assert output["status"]=="descriptive_only"
    assert output["frame"].churn_rate.notna().all()
    assert output["frame"].p_value.isna().all() and output["frame"].primary_ci95_lower.isna().all()


def test_complete_small_artifact_export_contains_six_graphs_and_all_contract_tables(fixture,tmp_path):
    features,predictions,config,_=fixture
    result=build_decisions(features,predictions,config)
    obs,metadata=factual(features)
    strategy,meta=strategy_factual()
    output=write_outputs(result,tmp_path/"decision",qini=qini_curves(obs,{"fixture":predictions["uplift"]},metadata),
                         strategy=strategy_evaluation(strategy,meta))
    assert pd.read_csv(output/"customers.csv").columns.tolist()==CUSTOMER_COLUMNS
    assert len(list((output/"graphs").glob("*.png")))==6 and len(list((output/"graphs").glob("*.pdf")))==6
    manifest=json.loads((output/"manifest.json").read_text())
    assert manifest["status"]=="complete" and "graph_data/qini.csv" in manifest["file_sha256"]
    with pytest.raises(ValueError,match="already exists"):write_outputs(result,output)


def test_unavailable_graphs_are_labelled_not_flat_fake_lines(fixture,tmp_path):
    features,predictions,config,_=fixture
    predictions={key:value for key,value in predictions.items() if key!="gross_4w"}
    predictions["clv"].frame["predicted_clv_12m"]=np.nan
    predictions["uplift"].frame[["p0","p1"]]=np.nan
    output=write_outputs(build_decisions(features,predictions,config),tmp_path/"missing")
    assert json.loads((output/"graph_data/qini_status.json").read_text())["status"]=="unavailable"
    assert json.loads((output/"budget_summary.json").read_text())["expected_incremental_revenue"] is None


@pytest.mark.parametrize("human_profile", [False, True])
def test_cli_reads_checksummed_features_relative_batches_and_factual_evaluation(fixture,tmp_path,monkeypatch,human_profile):
    from final_strategy.__main__ import main
    from simulate.v4.time_spec import weekly_metadata
    features,predictions,config,_=fixture
    source=tmp_path/"features"
    source.mkdir()
    feature_path=source/"feature_store.parquet"
    features.to_parquet(feature_path,index=False)
    (source/"manifest.json").write_text(json.dumps({**weekly_metadata(),"file_sha256":{
        feature_path.name:hashlib.sha256(feature_path.read_bytes()).hexdigest()}}))
    batches={}
    for role,batch in predictions.items():
        batch.frame.to_csv(tmp_path/f"{role}.csv",index=False)
        batches[role]={"frame":f"{role}.csv","metadata":asdict(batch.metadata)}
    (tmp_path/"batches.json").write_text(json.dumps(batches))
    (tmp_path/"config.json").write_text(json.dumps(asdict(config)))
    (tmp_path/"qini_models.json").write_text(json.dumps({"purchase-model":batches["uplift"]}))
    obs,_=factual(features)
    obs.to_csv(tmp_path/"qini.csv",index=False)
    strategy,metadata=strategy_factual()
    strategy.to_csv(tmp_path/"strategy.csv",index=False)
    assigned_roster(strategy).to_csv(tmp_path/"assignments.csv",index=False)
    (tmp_path/"evaluation.json").write_text(json.dumps(metadata))
    output=tmp_path/"decisions"
    arguments=["final_strategy","--features",str(source),"--predictions",str(tmp_path/"batches.json"),
        "--config",str(tmp_path/"config.json"),"--qini-outcomes",str(tmp_path/"qini.csv"),
        "--qini-predictions",str(tmp_path/"qini_models.json"),"--strategy-outcomes",str(tmp_path/"strategy.csv"),
        "--strategy-assignments",str(tmp_path/"assignments.csv"),
        "--evaluation-metadata",str(tmp_path/"evaluation.json"),"--output",str(output)]
    if human_profile:
        declarations=dict(version="fixture",threshold_source="synthetic",common=".2",risk_high=".1",risk_low=".3",
                          high_value=".1",standard_value=".3",h4_r=dict(a=0,b=".1",c=".3",d=".4"))
        (tmp_path/"profiles.json").write_text(json.dumps(declarations))
        arguments += ["--profile-declarations",str(tmp_path/"profiles.json"),"--profile","H1"]
    monkeypatch.setattr(sys,"argv",arguments)
    main()
    assert pd.read_csv(output/"customers.csv").columns.tolist()==CUSTOMER_COLUMNS
    assert json.loads((output/"budget_summary.json").read_text())["send_count"]==2
    assert json.loads((output/"graph_data/strategy_status.json").read_text())["status"] == "available"
    if human_profile:
        assert json.loads((output/"metadata.json").read_text())["human_strategy_profile"]["policy_id"]=="H1"
        assert (output/"graph_data/human_profile.csv").is_file()
    assert json.loads((output/"graph_data/qini_status.json").read_text())["status"]=="available"
    assert json.loads((output/"graph_data/strategy_status.json").read_text())["comparison_family"]==[
        "B versus A: churn_rate","C versus A: churn_rate"]
    features.iloc[:1].to_parquet(feature_path,index=False)
    with pytest.raises(ValueError,match="checksum"):
        load_feature_store(source)
    manifest=json.loads((source/"manifest.json").read_text())
    manifest["time_windows_days"]["recent_days"]=30
    (source/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="time contract"):
        load_feature_store(source)
