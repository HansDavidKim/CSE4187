"""Independent raw/sidecar and descriptive-distribution checks on small records."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from simulate.v4 import validation as audit
from simulate.v4.simulator.raw_store import RAW_COLUMNS, save_raw_dataset, validate_raw_dataset


@pytest.mark.parametrize("explicit_model", [None, "explicit-checkout.json"])
def test_validation_cli_uses_the_dataset_source_model_unless_explicitly_overridden(tmp_path, monkeypatch, explicit_model):
    data = tmp_path / "data"
    data.mkdir()
    (data / "raw.manifest.json").write_text(json.dumps({"metadata": {"model_path": "recorded-checkout.json"}}))
    output = tmp_path / "validation.json"
    arguments = ["validate", "--data-dir", str(data), "--output", str(output)]
    if explicit_model:
        arguments.extend(["--model", explicit_model])
    monkeypatch.setattr(sys, "argv", arguments)
    seen = []
    def validate(directory, model):
        seen.append((directory, model))
        return {"exact_checks_passed": True}
    monkeypatch.setattr(audit, "validate_directory", validate)
    assert audit.main() == 0
    assert seen == [(data, Path(explicit_model or "recorded-checkout.json"))]


@pytest.fixture
def records():
    t0 = pd.Timestamp("2024-01-07 15:00", tz="UTC")  # Monday 00:00 in Seoul.
    tables = {name: pd.DataFrame(columns=columns) for name, columns in RAW_COLUMNS.items()}
    rows = {
        "customers": [dict(customer_id=uid, signed_up_at=t0-pd.Timedelta(days=45)) for uid in ("u1", "u2")],
        "products": [dict(product_id="A", category_id="c", unit_price_minor=5),
                     dict(product_id="B", category_id=None, unit_price_minor=10)],
        "sessions": [dict(session_id="h1", customer_id="u1", started_at=t0-pd.Timedelta(hours=2), ended_at=t0-pd.Timedelta(hours=1)),
                     dict(session_id="h2", customer_id="u2", started_at=t0-pd.Timedelta(hours=3), ended_at=t0-pd.Timedelta(hours=2)),
                     dict(session_id="f1", customer_id="u1", started_at=t0, ended_at=t0+pd.Timedelta(minutes=10)),
                     dict(session_id="boundary", customer_id="u2", started_at=t0+pd.Timedelta(days=30), ended_at=t0+pd.Timedelta(days=30, minutes=10))],
        "events": [dict(event_id=i, customer_id="u1", session_id="h1" if i<4 else "f1",
                        event_at=t0-pd.Timedelta(hours=2)+pd.Timedelta(seconds=i) if i<4 else t0,
                        event_type=kind, product_id=pid, quantity=quantity, unit_price_minor=price,
                        offer_product_id=None, offer_unit_price_minor=None)
                   for i,kind,pid,quantity,price in [(1,"cart","A",1,5),(2,"cart","B",1,10),
                                                    (3,"remove_from_cart","B",1,10),(4,"purchase",None,None,None)]],
        "orders": [dict(order_id="o1", purchase_event_id=4, checkout_source="cart", coupon_id="coupon1",
                        gross_minor=5, discount_minor=1, paid_minor=4)],
        "order_items": [dict(order_id="o1", line_no=1, product_id="A", quantity=1, unit_price_minor=5)],
        "coupons": [dict(coupon_id="coupon1", customer_id="u1", sent_at=t0, expires_at=t0+pd.Timedelta(days=30),
                         discount_rate=0.1, policy_id="once10", send_cost_minor=1)],
    }
    tables = {name: pd.DataFrame(rows[name], columns=columns) for name, columns in RAW_COLUMNS.items()}
    metadata = dict(dataset_id="validation-fixture", seed=7, currency="SOURCE_UNSPECIFIED", amount_unit="0.01",
                    as_of_at=t0.isoformat(), observation_start_at=(t0-pd.Timedelta(days=60)).isoformat(),
                    observation_end_at=(t0+pd.Timedelta(days=60)).isoformat())
    labels = pd.DataFrame([dict(customer_id=uid, as_of_at=t0, snapshot_at=t0, outcome_end_at=t0+pd.Timedelta(days=30),
                               churn=churn, retained=1-churn, future_retained=1-churn, future_revenue_30d=revenue,
                               future_revenue=revenue, future_revenue_12m=None, is_censored=False,
                               definition_id="future_no_session_30d_active30d")
                           for uid,churn,revenue in [("u1",0,4),("u2",1,0)]])
    treatments = pd.DataFrame([dict(customer_id=uid, as_of_at=t0, assigned_at=t0, treatment=arm,
                                   assignment_probability=0.5, actual_sent=bool(arm),
                                   coupon_id="coupon1" if arm else None, send_cost_minor=arm)
                               for uid,arm in [("u1",1),("u2",0)]])
    validate_raw_dataset(tables, metadata)
    return tables, metadata, labels, treatments


def test_exact_sidecars_use_raw_future_half_open_windows_and_money(records):
    tables, metadata, labels, treatments = records
    result = audit.audit_sidecars(tables, metadata, labels, treatments)
    assert result["passed"] and not any(result["mismatch_counts"].values())
    assert result["windows"][0]["recomputed_churn_rate"] == 0.5
    assert result["windows"][0]["ITT_arm_counts"] == {"0":1,"1":1}
    assert result["money_minor"] == dict(item_gross=5, order_gross=5, discount=1, paid=4)
    # metadata intentionally has observation_* only, without legacy start_at/end_at.
    assert "start_at" not in metadata


@pytest.mark.parametrize("mutation,field", [("label","churn_label"),("revenue","future_revenue_30d"),
    ("probability","assignment_probability"),("assignment_time","assignment_time"),("sent","actual_sent"),
    ("coupon_reference","treated_coupon_reference"),("definition","definition_id"),("censor","is_censored")])
def test_corrupted_sidecar_is_detected_without_latent_truth(records, mutation, field):
    tables, metadata, labels, treatments = records
    if mutation=="label": labels.loc[0,"churn"]=1
    elif mutation=="revenue": labels.loc[0,"future_revenue_30d"]=5
    elif mutation=="probability": treatments.loc[0,"assignment_probability"]=0.6
    elif mutation=="assignment_time": treatments.loc[0,"assigned_at"]+=pd.Timedelta(seconds=1)
    elif mutation=="sent": treatments.loc[0,"actual_sent"]=False
    elif mutation=="coupon_reference": treatments.loc[0,"coupon_id"]=None
    elif mutation=="definition": labels.loc[0,"definition_id"]="past_inactivity"
    elif mutation=="censor": labels.loc[0,"is_censored"]=True
    result=audit.audit_sidecars(tables,metadata,labels,treatments)
    assert not result["passed"] and result["mismatch_counts"][field]>0


def test_all_assigned_customers_are_kept_and_old_control_coupons_are_not_new_treatment(records):
    tables, metadata, labels, treatments = records
    with pytest.raises(ValueError,match="cover every customer"):
        audit.audit_sidecars(tables,metadata,labels.iloc[:1],treatments)
    with pytest.raises(ValueError,match="all treatment/control assignments"):
        audit.audit_sidecars(tables,metadata,labels,treatments.iloc[:1])
    previous=tables["coupons"].iloc[0].copy()
    previous["coupon_id"],previous["customer_id"]="prior-control","u2"
    previous["sent_at"]-=pd.Timedelta(days=10)
    tables["coupons"]=pd.concat([tables["coupons"],previous.to_frame().T],ignore_index=True)
    validate_raw_dataset(tables,metadata)
    assert audit.audit_sidecars(tables,metadata,labels,treatments)["passed"]


def test_cart_journal_matches_replay_and_is_customer_scoped(records,monkeypatch):
    tables,metadata,_,_=records
    replay=audit.cart_at
    def scoped_replay(scoped,uid,at):
        assert len(scoped["customers"])<=1
        assert set(scoped["events"].customer_id)<=set(scoped["customers"].customer_id)
        return replay(scoped,uid,at)
    monkeypatch.setattr(audit,"cart_at",scoped_replay)
    result=audit.audit_cart_sample(tables,metadata["as_of_at"],sample_size=1)
    assert result["passed"] and result["customers"]==1
    assert result["quantity"]["mean"]==1 and result["gross_minor"]["mean"]==5


@pytest.fixture
def mixed_cart_records(records):
    tables,metadata,_,_=records
    t0=pd.Timestamp(metadata["as_of_at"])
    metadata["raw_schema_version"]=2
    tables["sessions"].loc[tables["sessions"].session_id.eq("h1"),"ended_at"]=t0-pd.Timedelta(seconds=1)
    events=[]
    for event_id,seconds,kind,product,quantity,price in [
        (1,-7200,"cart","A",1,5),(2,-7200,"cart","A",1,5),
        (3,-7200,"cart","B",1,10),(4,-7200,"cart","B",1,10),(5,-7200,"cart","B",1,10),
        (6,-3600,"purchase",None,None,None),(7,-3600,"cart","A",1,5),
        (8,-1800,"purchase",None,None,None),(9,0,"purchase",None,None,None),
        (10,1,"cart","B",1,10),
    ]:
        events.append(dict(event_id=event_id,customer_id="u1",session_id="h1" if seconds<0 else "f1",
                           event_at=t0+pd.Timedelta(seconds=seconds),event_type=kind,product_id=product,
                           quantity=quantity,unit_price_minor=price,offer_product_id=None,offer_unit_price_minor=None))
    # Reverse storage order: ties must use event_id, and the boundary/future cannot leak in.
    tables["events"]=pd.DataFrame(events[::-1],columns=RAW_COLUMNS["events"])
    tables["orders"]=pd.DataFrame([
        dict(order_id=oid,purchase_event_id=eid,checkout_source=source,coupon_id=None,
             gross_minor=gross,discount_minor=0,paid_minor=gross)
        for oid,eid,source,gross in [("mixed",6,"mixed",25),("direct",8,"buy_now",40),("boundary",9,"cart",25)]
    ],columns=RAW_COLUMNS["orders"])
    tables["order_items"]=pd.DataFrame([
        dict(order_id=oid,line_no=line,product_id=product,quantity=quantity,unit_price_minor=price)
        for oid,line,product,quantity,price in [
            ("mixed",1,"A",1,5),("mixed",2,"A",2,5),("mixed",3,"B",1,10),
            ("direct",1,"A",2,5),("direct",2,"B",3,10),
            ("boundary",1,"A",1,5),("boundary",2,"B",2,10),
        ]
    ],columns=RAW_COLUMNS["order_items"])
    validate_raw_dataset(tables,metadata)
    return tables,metadata


def test_mixed_cart_journal_preserves_partial_inventory_and_excludes_boundary(mixed_cart_records,monkeypatch):
    tables,metadata=mixed_cart_records
    replay=audit.cart_at
    def scoped_replay(scoped,uid,at):
        assert scoped["events"].event_at.lt(pd.Timestamp(at)).all()
        assert set(scoped["orders"].order_id)=={"mixed","direct"}
        actual=replay(scoped,uid,at)
        assert actual=={"A":1,"B":2}
        return actual
    monkeypatch.setattr(audit,"cart_at",scoped_replay)
    result=audit.audit_cart_sample(tables,metadata["as_of_at"],sample_size=1)
    assert result["passed"] and result["mismatches"]==0
    assert result["quantity"]["mean"]==3 and result["gross_minor"]["mean"]==25


def test_mixed_cart_expected_journal_independently_detects_wrong_replay(mixed_cart_records,monkeypatch):
    tables,metadata=mixed_cart_records
    # A replay that forgets the held B consumed by mixed checkout must disagree.
    monkeypatch.setattr(audit,"cart_at",lambda *args: {"A":1,"B":3})
    result=audit.audit_cart_sample(tables,metadata["as_of_at"],sample_size=1)
    assert not result["passed"] and result["mismatches"]==1


@pytest.fixture
def coupon_response_records(records):
    from simulate.v4.simulator.coupon_response import sample_coupon_response
    tables,metadata,_,treatments=records
    t0=pd.Timestamp(metadata["as_of_at"])
    # An exactly-28-day-old order is historical; u1's order exactly at t0 is future.
    tables["events"]["event_id"]+=1
    tables["orders"]["purchase_event_id"]+=1
    prior=t0-pd.Timedelta(days=28)
    tables["sessions"]=pd.concat([tables["sessions"],pd.DataFrame([dict(session_id="prior",customer_id="u2",
        started_at=prior,ended_at=prior)])],ignore_index=True)
    tables["events"]=pd.DataFrame([dict(event_id=1,customer_id="u2",session_id="prior",event_at=prior,
        event_type="purchase",product_id=None,quantity=None,unit_price_minor=None,
        offer_product_id=None,offer_unit_price_minor=None),*tables["events"].to_dict("records")],columns=RAW_COLUMNS["events"])
    tables["orders"]=pd.concat([tables["orders"],pd.DataFrame([dict(order_id="prior",purchase_event_id=1,
        checkout_source="buy_now",coupon_id=None,gross_minor=5,discount_minor=0,paid_minor=5)])],ignore_index=True)
    tables["order_items"]=pd.concat([tables["order_items"],pd.DataFrame([dict(order_id="prior",line_no=1,
        product_id="A",quantity=1,unit_price_minor=5)])],ignore_index=True)
    model=dict(schema_version=1,type="tianchi_o2o_descriptive_response",inactivity_days=28,window_days=30,
               discount_rate=.1,groups={name:dict(receipts=1,redeemed=1,response_probability=1.,redemption_delay_days=[0])
                   for name in ("pooled","no_observed_purchase","purchased_within_28d","lapsed_purchase")})
    model["groups"]["no_observed_purchase"].update(receipts=0,redeemed=0,response_probability=0.,redemption_delay_days=[])
    rows=[]
    for uid,last,treatment in [("u1",None,1),("u2",prior,0)]:
        row=sample_coupon_response(model,metadata["seed"],uid,t0,last)
        row.update(treatment=treatment,coupon_id="coupon1" if treatment else None,response_status="redeemed" if treatment else "control")
        if treatment:
            row.update(opportunity_at=t0,baseline_action="cart",purchase_event_id=5,promoted_purchase=True,coupon_used=True)
        rows.append(row)
    validate_raw_dataset(tables,metadata)
    return tables,metadata,pd.DataFrame(rows),treatments,model


def test_coupon_responses_use_strict_history_and_preserve_optional_old_datasets(coupon_response_records):
    tables,metadata,diagnostic,treatments,model=coupon_response_records
    report=audit.audit_coupon_responses(tables,metadata,diagnostic,treatments,model)
    assert report["passed"] and report["coupon_used"]==1
    assert report["history_cohort_counts"]=={"no_observed_purchase":1,"purchased_within_28d":1}
    assert diagnostic.loc[0,"source_cohort"]=="pooled"
    assert audit.audit_coupon_responses(tables,metadata,None,treatments)["status"]=="not_present"
    # Crossing the exact 28-day boundary changes only the observed-history cohort.
    tables["events"].loc[tables["events"].event_id.eq(1),"event_at"]-=pd.Timedelta(seconds=1)
    tables["sessions"].loc[tables["sessions"].session_id.eq("prior"),["started_at","ended_at"]]-=pd.Timedelta(seconds=1)
    diagnostic.loc[1,"history_cohort"]="lapsed_purchase"
    diagnostic.loc[1,"source_cohort"]="lapsed_purchase"
    validate_raw_dataset(tables,metadata)
    assert audit.audit_coupon_responses(tables,metadata,diagnostic,treatments,model)["passed"]


def test_audit_recovers_scaled_coupon_odds_and_natural_checkout(coupon_response_records):
    tables, metadata, diagnostic, treatments, model = coupon_response_records
    for name in ("pooled", "purchased_within_28d"):
        model["groups"][name].update(receipts=2, response_probability=.5)
    diagnostic["response_probability"] = .99
    diagnostic.loc[0, ["baseline_action", "promoted_purchase"]] = ["purchase", False]
    metadata["coupon_effect_calibration"] = {"redemption_odds_multiplier": 99.}
    assert audit.audit_coupon_responses(tables, metadata, diagnostic, treatments, model)["passed"]
    forced = diagnostic.copy()
    forced.loc[0, ["baseline_action", "promoted_purchase"]] = ["view", True]
    report = audit.audit_coupon_responses(tables, metadata, forced, treatments, model)
    assert not report["passed"] and report["mismatch_counts"]["forced_purchase"] == 1
    metadata["coupon_effect_calibration"]["redemption_odds_multiplier"] = 1.
    report = audit.audit_coupon_responses(tables, metadata, diagnostic, treatments, model)
    assert not report["passed"] and report["mismatch_counts"]["response_probability"] == 2


def test_audit_distinguishes_new_checkout_policy_from_existing_outputs(coupon_response_records):
    tables, metadata, diagnostic, treatments, model = coupon_response_records
    assert audit.audit_coupon_responses(tables, metadata, diagnostic, treatments, model)["passed"]
    metadata["coupon_purchase_response"] = {"requires_policy_purchase": True}
    report = audit.audit_coupon_responses(tables, metadata, diagnostic, treatments, model)
    assert not report["passed"] and report["mismatch_counts"]["forced_purchase"] == 1


@pytest.mark.parametrize("mutation,field",[("clock","used_response_clock"),("fk","used_purchase_event_fk"),
    ("cohort","history_cohort"),("uniform","response_uniform"),("unused","used_order_count"),
    ("promotion","promotion_flag"),("control","unselected_or_control_used"),
    ("unselected","unselected_or_control_used")])
def test_coupon_response_corruption_is_detected(coupon_response_records,mutation,field):
    tables,metadata,diagnostic,treatments,model=coupon_response_records
    if mutation=="clock":
        model["groups"]["pooled"]["redemption_delay_days"]=[1]
        diagnostic.loc[0,"source_delay_days"]=1
        diagnostic.loc[0,"response_ready_at"]+=pd.Timedelta(days=1)
    elif mutation=="fk": diagnostic.loc[0,"purchase_event_id"]=999
    elif mutation=="cohort": diagnostic.loc[0,"history_cohort"]="purchased_within_28d"
    elif mutation=="uniform": diagnostic.loc[0,"response_uniform"]=1.
    elif mutation=="unused": diagnostic.loc[0,"coupon_used"]=False
    elif mutation=="promotion": diagnostic.loc[0,"baseline_action"]="purchase"
    elif mutation=="control": diagnostic.loc[1,["coupon_used","purchase_event_id"]]=[True,5]
    elif mutation=="unselected":
        model["groups"]["pooled"].update(redeemed=0,response_probability=0.,redemption_delay_days=[])
        diagnostic.loc[0,["response_probability","response_selected","source_delay_days","response_ready_at"]]=[0.,False,None,pd.NaT]
    report=audit.audit_coupon_responses(tables,metadata,diagnostic,treatments,model)
    assert not report["passed"] and report["mismatch_counts"][field]>0


def test_coupon_response_requires_all_assignment_rows(coupon_response_records):
    tables,metadata,diagnostic,treatments,model=coupon_response_records
    with pytest.raises(ValueError,match="cover every customer"):
        audit.audit_coupon_responses(tables,metadata,diagnostic.iloc[:1],treatments,model)


def test_declared_coupon_response_model_requires_diagnostics(records):
    tables,metadata,_,treatments=records
    metadata["coupon_purchase_response"]={"model_path":"declared-response.json"}
    with pytest.raises(ValueError,match="diagnostics are required when a response model is declared"):
        audit.audit_coupon_responses(tables,metadata,None,treatments)


def test_removal_followup_excludes_censoring_and_exact_upper_boundary():
    t0=pd.Timestamp("2024-01-01",tz="UTC")
    events=pd.DataFrame([dict(customer_id="u",product_id=product,event_at=t0+pd.Timedelta(seconds=seconds),
                             event_id=index,event_type=kind)
                         for index,product,kind,seconds in [(1,"A","remove_from_cart",0),
                             (2,"B","remove_from_cart",0),(3,"B","purchase",7*86400-1),
                             (4,"A","cart",7*86400),(5,"A","remove_from_cart",29*86400)]])
    result=audit.removal_transitions(events,t0+pd.Timedelta(days=30))
    seven=result["horizons"]["7"]
    assert seven["eligible"]==2 and seven["right_censored"]==1
    assert seven["later_cart"]==0 and seven["later_purchase"]==1
    assert result["horizons"]["30"]["later_cart"]==1


def test_metrics_keep_customer_session_keys_and_nullable_source_prices():
    t0=pd.Timestamp("2024-01-01",tz="UTC")
    visits=pd.DataFrame([dict(customer_id=uid,session_id="same",started_at=t0,ended_at=t0+pd.Timedelta(seconds=1)) for uid in ("u1","u2")])
    interactions=pd.DataFrame([dict(customer_id=uid,session_id="same",event_at=t0,event_id=i,event_type="purchase",
                                    product_id="A",category_id=None,quantity=1,unit_price_minor=None)
                               for i,uid in enumerate(("u1","u2"),1)])
    interactions["unit_price_minor"]=interactions.unit_price_minor.astype("Int64")
    arrays,weights,summary=audit._metric_arrays(interactions,visits,t0,t0+pd.Timedelta(days=1))
    assert arrays["session_item_proxy_length"].tolist()==[1,1]
    distance=audit._distance(arrays["purchased_price_minor"],[5],"minor",weights,[1])
    assert distance["source"]["records"]==0 and distance["wasserstein"] is None
    json.dumps({"summary":summary,"distance":distance},allow_nan=False)


@pytest.fixture
def directory(records,tmp_path):
    tables,metadata,labels,treatments=records
    t0=pd.Timestamp(metadata["as_of_at"])
    source=[]
    for user in (0,20,40,60,80):
        for day in (-10,-1,1):
            for second,kind in enumerate(("view","cart","purchase")):
                source.append(dict(event_time=(t0+pd.Timedelta(days=day,seconds=second)).isoformat(),
                                   event_type=kind,product_id="A",category_id="c",category_code="",brand="",
                                   price="0.05",user_id=str(user),user_session=f"{user}:{day}"))
    raw_folder=tmp_path/"raw-source"
    raw_folder.mkdir()
    source_path=raw_folder/"2024-01.csv"
    pd.DataFrame(source).to_csv(source_path,index=False)
    stat=source_path.stat()
    artifact=dict(seed=7,currency="SOURCE_UNSPECIFIED",minor_unit="0.01",
                  observation_start=(t0-pd.Timedelta(days=11)).isoformat(),
                  observation_end=(t0+pd.Timedelta(days=2)).isoformat(),
                  validation=dict(cutoff_at_utc=t0.isoformat()),
                  source=dict(sample_modulus=20,files=[dict(path=str(source_path.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)]))
    model=tmp_path/"model.json"
    model.write_text(json.dumps(artifact))
    metadata["model_sha256"]=hashlib.sha256(model.read_bytes()).hexdigest()
    dataset=tmp_path/"dataset"
    dataset.mkdir()
    save_raw_dataset(dataset/"raw.sqlite",tables,metadata)
    labels.to_parquet(dataset/"labels.parquet",index=False)
    treatments.to_parquet(dataset/"treatments.parquet",index=False)
    return dataset,model


def test_directory_and_cli_report_exact_checks_and_descriptive_holdouts(directory,tmp_path):
    dataset,model=directory
    report=audit.validate_directory(dataset,model)
    assert report["exact_checks_passed"]
    comparisons=report["distribution_comparison"]
    assert set(comparisons["source_scopes"])=={"all_sampled_source","user_holdout","time_holdout","user_and_time_holdout"}
    assert set(comparisons["generated_scopes"])=={"all_observed","pre_snapshot_30d","outcome_30d"}
    numeric=comparisons["generated_scopes"]["all_observed"]["comparisons"]["all_sampled_source"]["numeric_distances"]
    assert all(value["pass_threshold"] is None for value in numeric.values())
    output=tmp_path/"audit.json"
    process=subprocess.run([sys.executable,"-m","simulate.v4.validation","--data-dir",str(dataset),
                            "--model",str(model),"--output",str(output)],capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert json.loads(output.read_text())["exact_checks_passed"]


@pytest.mark.parametrize("case", ["changed", "missing"])
def test_directory_validation_checks_all_declared_sidecars(directory, case):
    dataset, model = directory
    diagnostics = dataset / "diagnostics.parquet"
    pd.DataFrame({"recorded": [True]}).to_parquet(diagnostics, index=False)
    metadata = json.loads((dataset / "raw.manifest.json").read_text())["metadata"]
    metadata["sidecar_sha256"] = {"diagnostics": hashlib.sha256(diagnostics.read_bytes()).hexdigest()}
    (dataset / "manifest.json").write_text(json.dumps(metadata))
    if case == "changed":
        pd.DataFrame({"recorded": [False]}).to_parquet(diagnostics, index=False)
    else:
        diagnostics.unlink()
    with pytest.raises(ValueError, match="diagnostics sidecar checksum"):
        audit.validate_directory(dataset, model)


def test_directory_loads_optional_coupon_responses_and_checks_model_sha(directory,coupon_response_records,tmp_path):
    original,shopping_model=directory
    tables,metadata,diagnostic,treatments,response_model=coupon_response_records
    response_path=tmp_path/"response.json"
    response_path.write_text(json.dumps(response_model))
    metadata.update(model_sha256=hashlib.sha256(shopping_model.read_bytes()).hexdigest(),coupon_purchase_response={
        "model_path":str(response_path),"model_sha256":hashlib.sha256(response_path.read_bytes()).hexdigest()})
    folder=tmp_path/"coupon-dataset"
    folder.mkdir()
    save_raw_dataset(folder/"raw.sqlite",tables,metadata)
    pd.read_parquet(original/"labels.parquet").to_parquet(folder/"labels.parquet",index=False)
    treatments.to_parquet(folder/"treatments.parquet",index=False)
    diagnostic.to_parquet(folder/"coupon_response_diagnostics.parquet",index=False)
    report=audit.validate_directory(folder,shopping_model)
    assert report["exact_checks_passed"] and report["coupon_response_diagnostics"]["status"]=="checked"
    response_path.write_text(response_path.read_text()+"\n")
    with pytest.raises(ValueError,match="coupon response model SHA256"):
        audit.validate_directory(folder,shopping_model)


def test_provenance_count_and_model_identity_are_not_silently_truncated(directory,monkeypatch):
    dataset,model=directory
    original=audit.load_source
    def extra_source(*args):
        rows,provenance=original(*args)
        provenance["files"].append(copy.deepcopy(provenance["files"][0]))
        return rows,provenance
    monkeypatch.setattr(audit,"load_source",extra_source)
    with pytest.raises(ValueError,match="file count"):
        audit.validate_directory(dataset,model)
    model.write_text(model.read_text()+"\n")
    with pytest.raises(ValueError,match="SHA256"):
        audit.validate_directory(dataset,model)


def test_declared_return_artifact_is_checked_when_present(directory,tmp_path):
    dataset,model=directory
    return_path=tmp_path/"return.json"
    return_path.write_text('{"fixture": true}')
    manifest_path=dataset/"raw.manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["metadata"].update(return_model_path=str(return_path),
                                return_model_sha256=hashlib.sha256(return_path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest))
    assert audit.validate_directory(dataset,model)["return_model_provenance"]["status"]=="verified"
    return_path.write_text('{"fixture": false}')
    with pytest.raises(ValueError,match="return model SHA256"):
        audit.validate_directory(dataset,model)


def test_28_day_outcome_keeps_30_day_revenue_eligibility_and_coupon_policy(records):
    tables,metadata,labels,treatments=records
    t0=pd.Timestamp(metadata["as_of_at"])
    # A return exactly on day 28 is outside four weeks, but inside the source 30-day result.
    tables["sessions"].loc[3,"started_at"]=t0+pd.Timedelta(days=28)
    tables["sessions"].loc[3,"ended_at"]=t0+pd.Timedelta(days=28,minutes=10)
    late=pd.DataFrame([dict(event_id=5,customer_id="u2",session_id="boundary",event_at=t0+pd.Timedelta(days=29),
                            event_type="purchase",product_id=None,quantity=None,unit_price_minor=None,
                            offer_product_id=None,offer_unit_price_minor=None)],columns=RAW_COLUMNS["events"])
    tables["sessions"].loc[3,"ended_at"]=t0+pd.Timedelta(days=29,minutes=10)
    tables["events"]=pd.DataFrame([*tables["events"].to_dict("records"),*late.to_dict("records")],columns=RAW_COLUMNS["events"])
    tables["orders"]=pd.concat([tables["orders"],pd.DataFrame([dict(order_id="late",purchase_event_id=5,
        checkout_source="buy_now",coupon_id=None,gross_minor=5,discount_minor=0,paid_minor=5)])],ignore_index=True)
    tables["order_items"]=pd.concat([tables["order_items"],pd.DataFrame([dict(order_id="late",line_no=1,
        product_id="A",quantity=1,unit_price_minor=5)])],ignore_index=True)
    labels.loc[1,"future_revenue_30d"]=5
    labels["future_revenue_28d"]=labels.future_revenue
    labels["outcome_end_at"]=t0+pd.Timedelta(days=28)
    labels["definition_id"]="future_no_session_28d_active30d"
    metadata.update(definition_id="future_no_session_28d_active30d",outcome_window_days=28,
                    generation_window_days=30,eligibility_window_days=30,
                    generation_definition_id="future_no_session_30d_active30d",parent_dataset_id=metadata["dataset_id"])
    validate_raw_dataset(tables,metadata)
    report=audit.audit_sidecars(tables,metadata,labels,treatments)
    assert report["passed"]
    assert report["windows"][0]["outcome_window_days"]==28
    assert report["windows"][0]["recomputed_churn_rate"]==0.5
    assert not report["mismatch_counts"]["experiment_coupon_time"]
    labels.loc[1,"future_revenue_30d"]=0
    broken=audit.audit_sidecars(tables,metadata,labels,treatments)
    assert not broken["passed"] and broken["mismatch_counts"]["future_revenue_30d"]==1


@pytest.mark.parametrize("experiment_days", [28, 30])
def test_real_outcome_view_helper_matches_validator_and_paired_marginal_contract(records, experiment_days):
    from simulate.v4.simulator.outcomes import redefine_outcomes
    from simulate.v4.simulator.paired_evaluation import paired_retention_rollouts
    tables,metadata,labels,treatments=records
    metadata.update(schema_version=7, simulator_version="source-shopping-v2",
                    definition_id=f"future_no_session_30d_active{experiment_days}d",
                    eligibility_window_days=experiment_days, coupon_validity_days=experiment_days)
    t0 = pd.Timestamp(metadata["as_of_at"])
    tables["coupons"]["expires_at"] = t0 + pd.Timedelta(days=experiment_days)
    tables["sessions"].loc[0, "started_at"] = t0 - pd.Timedelta(days=experiment_days)
    original={name:frame.copy(deep=True) for name,frame in tables.items()}
    diagnostics=pd.DataFrame(dict(customer_id=["u1","u2"],source_profile_id=["donor1","donor2"],home_persona=["g","g"],
                                  no_coupon_return_probability_30d=[0.8,0.8],coupon_return_probability_30d=[0.9,0.7]))
    model=dict(profiles=[dict(profile_id="donor1",home_group="g",idle_gap_seconds=[29*86400]),
                         dict(profile_id="donor2",home_group="g",idle_gap_seconds=[86400])],home_groups={"g":{}})
    view,view_metadata=redefine_outcomes({**tables,"labels":labels,"treatments":treatments,"diagnostics":diagnostics},
                                         metadata,outcome_days=28,shopping_model=model)
    assert audit.audit_sidecars(view,view_metadata,view["labels"],view["treatments"])["passed"]
    pairs,truth=paired_retention_rollouts(view["diagnostics"],view_metadata,repetitions=2)
    assert truth.set_index("customer_id").loc["u1","true_retention_uplift"]==pytest.approx(0.1*(28/30))
    assert pairs.outcome_window_days.eq(28).all()
    assert pairs.eligibility_window_days.eq(experiment_days).all()
    for name in RAW_COLUMNS:
        pd.testing.assert_frame_equal(view[name],original[name])
    tables["sessions"].loc[0, "started_at"] -= pd.Timedelta(nanoseconds=1)
    tables["coupons"].loc[0, "expires_at"] += pd.Timedelta(days=1)
    failed = audit.audit_sidecars(view, view_metadata, view["labels"], view["treatments"])
    assert failed["mismatch_counts"][f"not_eligible_active{experiment_days}d"] == 1
    assert failed["mismatch_counts"]["experiment_coupon_time"] == 1
