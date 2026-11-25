"""Small, matched-window simulator fidelity benchmark; never fits downstream models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


from simulate.v4.validation import (
    _distance, _generated_interactions, _metric_arrays, _session_completion, _source_interactions, _source_sessions, _summary,
)
from simulate.v4.simulator.engine import EcommerceSimulator
from simulate.v4.simulator.outcomes import experiment_window_days
from simulate.v4.configuration import DEFAULT_CONFIG, load_config


ACTIONS = ("view", "cart", "remove_from_cart", "purchase")
STATES = ("start", *ACTIONS, "end")


def source_home_groups(model, source):
    by_profile = {row["profile_id"]:row["home_group"] for row in model["profiles"]}
    return {uid:by_profile.get(hashlib.sha256(f"cosmetics:{uid}".encode()).hexdigest()[:16],"unmapped")
            for uid in source.user_id.unique()}


def source_cache(model, path):
    """Use the same immutable, provenance-bound source cache as model fitting."""
    from .irl.fit_donor_prior import load_prior_source
    rows, descriptor = load_prior_source(model, path)
    return rows, {**descriptor, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def transition_matrix(interactions, sessions, lower, upper, session_diagnostics=None):
    """Keep observed prefix transitions; append END only with completion support."""
    visits, completion = _session_completion(sessions, lower, upper, session_diagnostics)
    keys = pd.MultiIndex.from_frame(visits[["customer_id", "session_id"]])
    scoped = interactions.loc[interactions.event_at.ge(lower) & interactions.event_at.lt(upper)
                              & interactions.event_type.ne("coupon_usage")].copy()
    scoped = scoped.loc[pd.MultiIndex.from_frame(scoped[["customer_id", "session_id"]]).isin(keys)].reset_index(drop=True)
    if not scoped.event_type.isin(ACTIONS).all():
        raise ValueError("unknown transition action")
    quantity = pd.to_numeric(scoped.quantity, errors="raise")
    if quantity.isna().any() or quantity.lt(1).any() or quantity.ne(np.floor(quantity)).any():
        raise ValueError("transition item weights must be positive integers")
    scoped["unit_order"] = np.arange(len(scoped))
    scoped = scoped.loc[scoped.index.repeat(quantity.astype(int))].sort_values(
        ["customer_id", "session_id", "event_at", "event_id", "unit_order"], kind="stable")
    actions = scoped.groupby(["customer_id", "session_id"], sort=False).event_type.agg(list).to_dict()
    edges = []
    complete_keys = set(map(tuple, visits.loc[visits._complete, ["customer_id", "session_id"]].to_numpy()))
    for row in visits.itertuples(index=False):
        sequence = ["start", *actions.get((row.customer_id, row.session_id), [])]
        if (row.customer_id, row.session_id) in complete_keys:
            sequence.append("end")
        edges.extend((row.customer_id, before, after) for before, after in zip(sequence, sequence[1:]))
    edge_frame = pd.DataFrame(edges, columns=["customer_id", "from", "to"])
    counts = pd.crosstab(edge_frame["from"], edge_frame["to"]).reindex(index=STATES, columns=STATES, fill_value=0)
    probabilities = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    if len(edge_frame):
        edge_frame["weight"] = 1/edge_frame.groupby("customer_id").customer_id.transform("size")
        balanced = edge_frame.pivot_table(index="from", columns="to", values="weight", aggfunc="sum", fill_value=0).reindex(index=STATES, columns=STATES, fill_value=0)
        balanced = balanced.div(balanced.sum(axis=1).replace(0,np.nan),axis=0)
    else:
        balanced = probabilities.copy()
    def serial(frame):
        return {before:{after:(None if pd.isna(value) else float(value)) for after,value in row.items()}
                for before,row in frame.iterrows()}
    return {"states":list(STATES),"counts":{before:{after:int(value) for after,value in row.items()} for before,row in counts.iterrows()},
            "row_probabilities":serial(probabilities),"customer_balanced_row_probabilities":serial(balanced),
            "complete_sessions":len(complete_keys),"item_units":len(scoped),"edges":len(edges),
            "empty_sessions":sum(key not in actions for key in complete_keys),
            "session_completion":completion,"end_meaning":completion["basis"]}


def transition_distance(source, generated):
    rows = {}
    weighted = weight = 0.0
    for before in STATES[:-1]:
        left,right = source["row_probabilities"][before],generated["row_probabilities"][before]
        if any(value is None for value in left.values()) or any(value is None for value in right.values()):
            rows[before] = None
            continue
        distance = sum(abs(left[action]-right[action]) for action in STATES)
        count = sum(source["counts"][before].values())
        rows[before] = distance
        weighted += distance*count
        weight += count
    return {"row_L1":rows,"source_event_weighted_row_L1":weighted/weight if weight else None,
            "interpretation":"descriptive distance, not a pass threshold"}


def scope_metrics(interactions, sessions, ids, groups, lower, upper, as_of, session_diagnostics=None):
    ids = set(ids)
    events = interactions.loc[interactions.customer_id.isin(ids) & interactions.event_type.ne("coupon_usage")]
    visits = sessions.loc[sessions.customer_id.isin(ids)]
    arrays,price_weights,summary = _metric_arrays(events,visits,lower,upper,session_diagnostics)
    inside = visits.loc[visits.started_at.ge(lower) & visits.started_at.lt(upper)]
    array_ids = sorted(ids)
    arrays["sessions_per_selected_customer"] = inside.groupby("customer_id").size().reindex(array_ids,fill_value=0).to_numpy()
    bought = events.loc[events.event_at.ge(lower) & events.event_at.lt(upper) & events.event_type.eq("purchase")]
    arrays["purchase_sessions_per_selected_customer"] = bought.groupby("customer_id").session_id.nunique().reindex(array_ids,fill_value=0).to_numpy()
    first = visits.groupby("customer_id").started_at.min().reindex(array_ids)
    arrays["observed_tenure_days"] = (upper-first).dt.total_seconds().to_numpy()/86400
    arrays["observed_history_days_at_asof"] = (as_of-first).dt.total_seconds().to_numpy()/86400
    summary["selected_customers_including_zero_activity"] = len(ids)
    summary["observed_tenure_meaning"] = "first observed session, not verified source signup"
    summary["home_group_counts"] = {str(key):int(value) for key,value in pd.Series(
        [groups.get(uid,"unmapped") for uid in array_ids], dtype=object).value_counts().items()}
    for name in ("observed_tenure_days","observed_history_days_at_asof","sessions_per_selected_customer","purchase_sessions_per_selected_customer"):
        summary[name] = _summary(arrays[name])
    summary["transitions"] = transition_matrix(events,visits,lower,upper,session_diagnostics)
    return arrays,price_weights,summary


def compare_scope(source_events,source_sessions,source_ids,source_groups,generated,generated_sessions,generated_ids,generated_groups,lower,upper,as_of,session_diagnostics=None):
    left,lw,ls = scope_metrics(source_events,source_sessions,source_ids,source_groups,lower,upper,as_of)
    right,rw,rs = scope_metrics(generated,generated_sessions,generated_ids,generated_groups,lower,upper,as_of,session_diagnostics)
    distances = {}
    for name in left:
        unit = "minor_source_unit" if name.endswith("minor") else "seconds" if name.endswith("seconds") else "days" if "days" in name else "count"
        distances[name] = _distance(left[name],right[name],unit,lw if name=="purchased_price_minor" else None,rw if name=="purchased_price_minor" else None)
    for summary,events,visits,ids,groups,diagnostics in ((ls,source_events,source_sessions,source_ids,source_groups,None),(rs,generated,generated_sessions,generated_ids,generated_groups,session_diagnostics)):
        summary["home_group_strata"] = home_group_strata(events,visits,ids,groups,lower,upper,as_of,diagnostics)
    action_delta = {action:rs["item_proxy_action_share"].get(action,0)-ls["item_proxy_action_share"].get(action,0) for action in ACTIONS}
    return {"source":ls,"generated":rs,"numeric_distances":distances,
            "action_share_difference":action_delta,"action_share_L1":sum(abs(value) for value in action_delta.values()),
            "transition_distance":transition_distance(ls["transitions"],rs["transitions"])}


def home_group_strata(events,visits,ids,groups,lower,upper,as_of,session_diagnostics=None):
    strata = {}
    for group in sorted(set(groups.get(uid,"unmapped") for uid in ids)):
        selected = {uid for uid in ids if groups.get(uid,"unmapped")==group}
        arrays,weights,part = scope_metrics(events,visits,selected,groups,lower,upper,as_of,session_diagnostics)
        strata[group] = {"customers":len(selected),
            "action_share":part["item_proxy_action_share"],"transitions":part["transitions"],
            "remove_then_later":part["remove_then_later"],
            "numeric":{name:_summary(values,weights if name=="purchased_price_minor" else None) for name,values in arrays.items()}}
    return strata


def scalar_distances(scope):
    result = {}
    for name in ("repeat_product_across_sessions",):
        left,right = (scope[domain][name]["share"] for domain in ("source","generated"))
        result[name] = abs(left-right) if left is not None and right is not None else None
    for days in ("7","30"):
        for outcome in ("cart_share","purchase_share"):
            left,right = (scope[domain]["remove_then_later"]["horizons"][days][outcome] for domain in ("source","generated"))
            result[f"remove_to_{outcome}_{days}d"] = abs(left-right) if left is not None and right is not None else None
    return result


def compare_reports(baseline,candidate):
    protocol_keys=("source_cache_fingerprint","customers","seeds","months","clock_start","clock_end","as_of_at",
                   "source_cache_sha256", "configuration_except_profile_sha256", "session_completion_definition")
    equal = {key:baseline["protocol"].get(key)==candidate["protocol"].get(key) for key in protocol_keys}
    equal["eligibility_window_days"] = (
        experiment_window_days(baseline["protocol"], "eligibility_window_days")
        == experiment_window_days(candidate["protocol"], "eligibility_window_days")
    )
    changes = []
    for before,after in zip(baseline["runs"],candidate["runs"]):
        for scope in before["scopes"]:
            if scope not in after["scopes"]:
                continue
            a,b=before["scopes"][scope],after["scopes"][scope]
            metrics={"action_share_L1":b["action_share_L1"]-a["action_share_L1"]}
            av,bv=a["transition_distance"]["source_event_weighted_row_L1"],b["transition_distance"]["source_event_weighted_row_L1"]
            metrics["transition_L1_delta"] = bv-av if av is not None and bv is not None else None
            for name,old in scalar_distances(a).items():
                new=scalar_distances(b)[name]
                metrics[name] = new-old if new is not None and old is not None else None
            for name,old in a["numeric_distances"].items():
                new=b["numeric_distances"][name]
                metrics[name] = {key:(new[key]-old[key] if new[key] is not None and old[key] is not None else None)
                                 for key in ("ks_statistic_unweighted_records","wasserstein")}
            changes.append({"seed":before["seed"],"scope":scope,"distance_deltas_candidate_minus_baseline":metrics,
                            "baseline_generated_group_counts":a["generated"]["home_group_counts"],
                            "candidate_generated_group_counts":b["generated"]["home_group_counts"]})
    return {"protocol_matches":equal,"distribution_comparison_eligible":all(equal.values()),
            "same_profile_artifact":baseline["profile_artifact_sha256"]==candidate["profile_artifact_sha256"],
            "distance_direction":"negative is closer; positive is worse; null lacks comparable support",
            "not_common_random_number_replay":"the engine's action-dependent global RNG can change donors and assignments despite equal seeds",
            "changes":changes}


def run_benchmark(config,model,source,source_descriptor,customers,seeds):
    source_events,source_sessions = _source_interactions(source),_source_sessions(source)
    source_groups = source_home_groups(model,source)
    config=copy.deepcopy(config)
    config["simulation"]["start_at"] = model["observation_start"]
    runs=[]
    engine_path=Path(inspect.getfile(EcommerceSimulator))
    before_sha=hashlib.sha256(engine_path.read_bytes()).hexdigest()
    for seed in seeds:
        started=time.perf_counter()
        result=EcommerceSimulator(config).generate(n_customers=customers,months=5,seed=seed)
        metadata=result.metadata
        lower,upper,asof = (pd.Timestamp(metadata[key]) for key in ("observation_start_at","observation_end_at","as_of_at"))
        if lower!=pd.Timestamp(model["observation_start"]) or upper!=pd.Timestamp(model["observation_end"]):
            raise ValueError("benchmark requires the same five-calendar-month source/generation span")
        eligibility_days = experiment_window_days(metadata, "eligibility_window_days")
        history_start = asof - pd.Timedelta(days=eligibility_days)
        active_source = set(source_sessions.loc[source_sessions.started_at.ge(history_start) & source_sessions.started_at.lt(asof),"customer_id"])
        gen_sessions=result.tables["sessions"]
        active_generated=set(gen_sessions.loc[gen_sessions.started_at.ge(history_start) & gen_sessions.started_at.lt(asof),"customer_id"])
        generated=_generated_interactions(result.tables)
        groups=result.tables["diagnostics"].set_index("customer_id").home_persona.to_dict()
        controls=set(result.tables["treatments"].loc[result.tables["treatments"].treatment.eq(0),"customer_id"]) & active_generated
        source_returned = set(source_sessions.loc[source_sessions.started_at.ge(asof) &
                              source_sessions.started_at.lt(asof + pd.Timedelta(days=28)), "customer_id"]) & active_source
        control_labels = result.tables["labels"].loc[result.tables["labels"].customer_id.isin(controls)]
        scopes={}
        for name,start,end,ids in (("matched_five_months_no_new_coupon",lower,upper,controls),
                                   (f"pre_asof_{eligibility_days}d",history_start,asof,active_generated),
                                   ("post_asof_28d_no_new_coupon",asof,asof+pd.Timedelta(days=28),controls)):
            scopes[name]=compare_scope(source_events,source_sessions,active_source,source_groups,generated,gen_sessions,ids,groups,start,end,asof,
                                       result.tables.get("session_diagnostics"))
        donor_ids=result.tables["diagnostics"].source_profile_id.astype(str)
        runs.append({"seed":seed,"seconds":time.perf_counter()-started,"dataset_id":metadata["dataset_id"],
                     "runtime_code_sha256": metadata["runtime_code_sha256"],
                     "configuration_sha256": metadata["configuration_sha256"],
                     "retention_mode": metadata.get("retention_mode", "target"),
                     "source_no_session_28d_rate": 1 - len(source_returned) / len(active_source),
                     "generated_control_no_session_28d_rate": float(control_labels.churn.mean()),
                     "history_logit_shift": metadata["history_logit_shift"], "outcome_logit_shift": metadata["outcome_logit_shift"],
                     "generator_rows":metadata["table_rows"],"canonical_order_headers":len(result.tables["orders"]),
                     "generated_customers":customers,"source_active_cohort_customers":len(active_source),
                     "generated_active_cohort_customers":len(active_generated),"generated_no_new_coupon_customers":len(controls),
                     "donor_draw_sha256":hashlib.sha256(donor_ids.to_json().encode()).hexdigest(),
                     "lifecycle_counts":result.tables["lifecycle"].life_cycle.value_counts().to_dict(),
                     "scopes":scopes})
    after_sha=hashlib.sha256(engine_path.read_bytes()).hexdigest()
    if before_sha!=after_sha:
        raise ValueError("engine changed during benchmark; freeze it for both seeds")
    comparable_config=copy.deepcopy(config)
    comparable_config["source_simulation"].pop("model_path", None)
    return {"schema_version":1,"purpose":"simulator fidelity only; no downstream fitting or existing dataset reads",
            "engine_sha256":before_sha,"profile_artifact_sha256":hashlib.sha256(Path(config["source_simulation"]["model_path"]).read_bytes()).hexdigest(),
            "return_artifact_sha256":hashlib.sha256(Path(config["source_simulation"]["return_model_path"]).read_bytes()).hexdigest() if config["source_simulation"].get("return_model_path") else None,
            "configuration": config,
            "protocol":{"source_cache_fingerprint":source_descriptor["fingerprint"],
                        "source_cache_sha256": source_descriptor.get("sha256"),
                        "session_completion_definition": "diagnostic_natural_end_or_source_30min_v1",
                        "configuration_except_profile_sha256": hashlib.sha256(json.dumps(comparable_config, sort_keys=True).encode()).hexdigest(),
                        "customers":customers,"seeds":seeds,"months":5,
                        "clock_start":lower.isoformat(),"clock_end":upper.isoformat(),"as_of_at":asof.isoformat(),
                        "eligibility_window_days":eligibility_days,
                        "eligibility":f"same session-start criterion [asof-{eligibility_days}d,asof) applied separately in both domains"},
            "source_cache":source_descriptor,"runs":runs,
            "interpretation":[
                "Same calendar span/asof/eligibility criterion; source and generated people and donor composition are not individually matched.",
                "Post-asof comparisons use generated no-new-coupon customers; source promotion exposure is unknown.",
                "Source first observation is left-censored, not signup; generated signup and empty initial cart are structural.",
                "Source purchase item rows use a one-unit proxy; generated OrderItems are expanded by quantity, with order headers reported separately.",
                "Lengths/durations use diagnostic natural END; censored prefixes retain observed transitions without an invented END.",
                "Source inferred END requires last action +30min strictly before the scope upper bound; legacy timestamp-only END is unverified.",
                "7d removal followup requires full observation; 30d rates in short scopes can be entirely right-censored.",
                "Idle gaps require the previous session's completed END and the next observed START; censored ends are never bridged or zero-filled.",
                "Source home groups use full observed histories and generated donor home groups are diagnostic strata, not verified innate personas.",
                "Distances describe fidelity without invented pass thresholds; subgroup sample sizes and composition changes remain visible."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--assets-root",type=Path)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--source-cache",type=Path,default=Path("artifacts/v4/source_sample.parquet"))
    parser.add_argument("--customers",type=int,default=200)
    parser.add_argument("--seeds",type=int,nargs="+",default=[17,29])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline",type=Path)
    args=parser.parse_args()
    if args.customers<2 or args.customers>5000 or args.customers%2 or len(set(args.seeds))!=len(args.seeds):
        parser.error("use an even customer count from 2 through 5000 and unique seeds")
    if args.output.exists():
        raise FileExistsError("choose a new benchmark JSON; existing reports are preserved")
    config=load_config(args.config,args.assets_root)
    if args.model is not None:
        config["source_simulation"]["model_path"]=str(args.model.resolve())
    model=json.loads(Path(config["source_simulation"]["model_path"]).read_text())
    source,descriptor=source_cache(model,args.source_cache)
    report=run_benchmark(config,model,source,descriptor,args.customers,args.seeds)
    report["analysis_script_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.baseline:
        report["baseline_comparison"]=compare_reports(json.loads(args.baseline.read_text()),report)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"output":str(args.output),"engine_sha256":report["engine_sha256"],"seeds":args.seeds,"customers":args.customers}))


if __name__=="__main__":
    main()
