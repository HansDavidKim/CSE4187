from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
from itertools import product
import json

import pytest

from final_strategy import budget as engine
from final_strategy.budget_quality_integration import certify_budget_result, certify_what_if_result
from final_strategy.tests.test_budget import population, DATASET, T0
from final_strategy.tests.test_integration import cohort, retain, assert_unchanged
from final_strategy.what_if import run_what_if
from final_strategy.validation import InputValidationError


def setup(scores=(60,100,120), costs=(10,20,30), amount=50, places=0, algorithm="greedy"):
    s,c=population(scores,costs)
    b=engine.Budget(amount,"SOURCE_UNSPECIFIED",".01")
    m=engine.CostMetadata("explicit","v1",places,T0)
    r=engine.optimize_budget(s,c,b,m,dataset_id=DATASET,as_of_at=T0,algorithm=algorithm)
    return s,c,b,m,r


def certify(args,**kw):
    return certify_budget_result(*args,dataset_id=DATASET,as_of_at=T0,**kw)


def test_posthoc_no_optimizer_calls_and_immutability(monkeypatch):
    args=setup(); original=deepcopy(args)
    def forbidden(*a,**kw): raise AssertionError("optimization repeated")
    monkeypatch.setattr(engine,"_execute",forbidden)
    monkeypatch.setattr(engine,"_solve_milp",forbidden)
    q=certify(args)
    assert q.certificate.lower_bound==160 and q.certificate.upper_bound==240
    assert q.certification_status=="feasible_uncertified"
    assert q.to_dict()["budget_compliant"] and not q.to_dict()["certified"]
    assert certify(args,epsilon="0.34").certification_status=="epsilon_certified"
    json.dumps(q.to_dict())
    assert_unchanged(args,original)


@pytest.mark.parametrize("scores,costs,amount,places", [([],[],0,0),([1,2],[0,0],0,0),
    ([-1,0,2],[1,0,1],0,0),([1,1],[1,1],1,0),([10,20,30],[".050000025"]*3,".1",9),
    ([1,2],[".000000001",".000000002"],".000000002",9),([2,4],[2,4],5,0),([2,3],[2,3],3,0)])
def test_boundaries(scores,costs,amount,places):
    q=certify(setup(scores,costs,amount,places))
    assert q.certificate.total_expected_cost<=Decimal(str(amount))
    assert q.certificate.lower_bound<=q.certificate.upper_bound


@pytest.mark.parametrize("seed",range(10))
def test_exhaustive(seed):
    import numpy as np
    rng=np.random.default_rng(seed)
    scores=rng.integers(1,40,7).tolist(); costs=rng.integers(0,10,7).tolist()
    q=certify(setup(scores,costs,15))
    optimum=max(sum(v*b for v,b in zip(scores,bits)) for bits in product((0,1),repeat=7)
                if sum(v*b for v,b in zip(costs,bits))<=15)
    assert q.certificate.lower_bound<=optimum<=q.certificate.upper_bound
    if q.to_dict()["certified"]: assert q.certificate.lower_bound>=(1-q.epsilon)*optimum


@pytest.mark.parametrize("change",["unknown","duplicate","binary","cost","snapshot","score","summary","budget","source","precision","exclusion"])
def test_invalid_result_binding(change):
    args=list(setup()); r=args[-1]; rows=r.results["greedy"].decisions
    if change=="unknown": rows.loc[0,"customer_id"]="unknown"
    elif change=="duplicate": rows.loc[0,"customer_id"]=rows.loc[1,"customer_id"]
    elif change=="binary": rows["selected"]=rows.selected.astype(float); rows.loc[0,"selected"]=.5
    elif change=="cost": rows.loc[0,"expected_cost"]=Decimal(1)
    elif change=="snapshot": rows.loc[0,"snapshot_id"]="wrong"
    elif change=="score": rows.loc[0,"priority_score"]=1
    elif change=="summary": r.results["greedy"].summary["objective_value"]=0
    elif change=="budget": args[2]=engine.Budget(51,"SOURCE_UNSPECIFIED",".01")
    elif change=="source": args[1].loc[0,"raw_database_sha256"]="wrong"
    elif change=="precision": args[3]=engine.CostMetadata("explicit","v1",1,T0)
    elif change=="exclusion": rows.loc[0,"algorithm"]="milp"
    with pytest.raises(InputValidationError): certify(args)


def test_missing_and_invalid_greedy():
    assert certify(setup(algorithm="milp")).certification_status=="unavailable"
    args=setup(); args[-1].results["greedy"].summary["has_valid_solution"]=False
    assert certify(args).reason=="greedy_result_invalid"


def test_actual_retention_and_what_if(cohort):
    retention=retain(cohort)
    s,c,m=retention.budget_inputs,retention.costs,retention.cost_metadata
    options=dict(dataset_id="dummy-segmentation",as_of_at=T0)
    b=cohort["budget"]
    r=engine.optimize_budget(s,c,b,m,algorithm="both",**options)
    w=run_what_if(s,c,b,m,algorithm="both",**options)
    original=deepcopy((retention,r,w))
    q=certify_budget_result(s,c,b,m,r,**options)
    assert q.metadata["input_customer_count"]==60 and q.metadata["candidate_count"]==18
    assert q.metadata["excluded_count"]==42 and q.certificate.lower_bound==400
    assert q.certificate.total_expected_cost==Decimal("9.00")
    certificates=certify_what_if_result(s,c,b,m,w,**options)
    assert [x.certificate.lower_bound for x in certificates.values()]==[200,400,800]
    assert all(x.certification_status=="proven_optimal" for x in certificates.values())
    assert len({x.metadata["selection_fingerprint"] for x in certificates.values()})==3
    assert_unchanged((retention,r,w),original)


@pytest.mark.parametrize("algorithm",["greedy","milp"])
def test_what_if_subset_and_missing_greedy(algorithm):
    s,c,b,m,_=setup(places=1)
    w=run_what_if(s,c,b,m,dataset_id=DATASET,as_of_at=T0,algorithm=algorithm,scenarios=["budget_100"])
    original=deepcopy(w)
    q=certify_what_if_result(s,c,b,m,w,dataset_id=DATASET,as_of_at=T0)
    assert list(q)==["budget_100"]
    assert (q["budget_100"].certificate is None)==(algorithm=="milp")
    assert_unchanged(w,original)


@pytest.mark.parametrize("field,value",[("dataset_id","wrong"),("as_of_at",T0.replace(year=2025)),("currency","USD"),("amount_unit","1")])
def test_result_context_mismatch(field,value):
    args=setup(); args[-1].results["greedy"].decisions.loc[0,field]=value
    with pytest.raises(InputValidationError): certify(args)


@pytest.mark.parametrize("field",["scenario_budgets","summary","decisions"])
def test_what_if_budget_binding(field):
    s,c,b,m,_=setup(places=1)
    w=run_what_if(s,c,b,m,dataset_id=DATASET,as_of_at=T0,algorithm="greedy")
    if field=="scenario_budgets": w.metadata[field]["budget_050"]["budget"]=Decimal(99)
    else: getattr(w,field).loc[0,"budget"]=Decimal(99)
    with pytest.raises(InputValidationError):
        certify_what_if_result(s,c,b,m,w,dataset_id=DATASET,as_of_at=T0)


def test_selected_overbudget_rejected_even_with_valid_summary_flag():
    args=setup([10,20,30],[".050000025"]*3,".1",9)
    args[-1].results["greedy"].decisions["selected"]=1
    with pytest.raises(InputValidationError): certify(args)
