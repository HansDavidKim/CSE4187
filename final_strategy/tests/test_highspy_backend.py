"""C-2 mocks, real Windows ownership tests, and separately identifiable optional highspy tests."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import sys

import pandas as pd
import pytest

from final_strategy import hybrid, budget_quality as quality, _solver_process as process
from final_strategy._highspy_worker import VERSION, IDENTITY, loads, validate_request, validate_response
from final_strategy.hybrid import HybridConfig
from final_strategy.tests.test_budget import run, population, chosen, DATASET, T0
from final_strategy.tests.test_hybrid import exact, what_if
from final_strategy.tests.test_budget_quality import problem
from final_strategy.what_if import export_what_if
from final_strategy.validation import InputValidationError

REAL = os.name == "nt" and importlib.util.find_spec("highspy") is not None


def request():
    return dict(protocol_version=VERSION, request_id="request-1", input_fingerprint="source-1",
        variable_order_fingerprint="order-1", n=3, customer_ids=["001", "002", "003"],
        scores_hex=[float(v).hex() for v in (60,100,120)], weights=[1,2,3], capacity=5, initial=[1,1,0],
        options=dict(time_limit=3.0, presolve=False, threads=1, mip_rel_gap=0.0, submit_initial=True))


def response(req, vector=(0,1,1), **changes):
    result={k:req[k] for k in IDENTITY}
    result.update(solver_status="Optimal", solver_message="mock", vector=list(vector), primal_objective=-1.,
        dual_bound=-1., mip_gap=0., node_count=1, solver_executed=True, initial_solution_submitted=True,
        initial_solution_api_status="HighsStatus.kOk", initial_solution_acceptance="unknown",
        initial_solution_acceptance_evidence=None, engine_version="mock",
        timings=dict(model_seconds=0., initial_solution_seconds=0., solve_seconds=0., worker_seconds=0.))
    result.update(changes)
    return result


def installed(monkeypatch, callback):
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda name: object())
    def worker(req, timeout):
        return dict(status="completed", child_created=True, solver_executed=True,
                    response=callback(req), progress=None)
    monkeypatch.setattr(process,"run_worker",worker)


def execute(**kwargs):
    return run(algorithm="hybrid", hybrid_config=HybridConfig(backend="highspy", local_improvement=False, **kwargs))


def test_none_and_certified_never_probe_backend(monkeypatch):
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda name: pytest.fail("backend probed"))
    baseline=run(algorithm="hybrid")
    assert "backend" not in baseline.metadata["hybrid"]
    certified=run(*population([1],[1]),amount=1,algorithm="hybrid",hybrid_config=HybridConfig(backend="highspy"))
    assert certified.metadata["hybrid"]["backend"]["reason"]=="quality_target_met"
    assert not certified.metadata["hybrid"]["additional_solver_used"]


def test_missing_dependency_preserves_c1(monkeypatch):
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda name: None)
    r=execute()
    assert chosen(r,"hybrid")=={"00000","00001"}
    assert r.metadata["hybrid"]["backend"]["status"]=="backend_not_installed"
    assert not r.metadata["hybrid"]["backend"]["child_created"]


@pytest.mark.parametrize("setting", [dict(backend="scip"),dict(solver_time_limit=-1),dict(solver_time_limit=float('nan')),
    dict(external_timeout=float('inf')),dict(external_timeout=-1),dict(mip_rel_gap=1.1),dict(mip_rel_gap=-.1),
    dict(presolve=1),dict(threads=True),dict(threads=0),dict(threads=65)])
def test_invalid_backend_config(setting):
    with pytest.raises(InputValidationError): HybridConfig(**setting)


def test_valid_improvement_uses_exact_parent_score_and_keeps_upper(monkeypatch):
    seen=[]
    def candidate(req):
        validate_request(req); seen.append(deepcopy(req))
        return response(req, primal_objective=-99999.)
    installed(monkeypatch,candidate)
    r=execute()
    assert seen[0]["initial"]==[1,1,0]
    assert [float.fromhex(s) for s in seen[0]["scores_hex"]]==[60,100,120]
    h=r.metadata["hybrid"]; c=h["quality"]["certificate"]
    assert exact(c["lower_bound"])==220 and exact(c["upper_bound"])==240
    assert r.results["hybrid"].summary["status"]=="feasible_uncertified"
    assert h["selection"]["source"]=="highspy" and h["backend"]["candidate_adopted"]
    assert h["backend"]["initial_solution_acceptance"]=="unknown"
    assert not r.results["hybrid"].summary["accepted_for_planning"]
    assert r.results["hybrid"].summary["optimality_gap"] is None


def test_local_best_is_warm_start(monkeypatch):
    seen=[]
    installed(monkeypatch,lambda req: (seen.append(req) or response(req)))
    r=run(algorithm="hybrid",hybrid_config=HybridConfig(backend="highspy"))
    assert seen[0]["initial"]==[0,1,1]
    assert not r.metadata["hybrid"]["backend"]["candidate_adopted"]
    assert r.metadata["hybrid"]["selection"]["source"]=="local_improvement"


def test_initial_rejection_does_not_discard_parent(monkeypatch):
    installed(monkeypatch,lambda req: response(req,vector=[1,0,0],initial_solution_api_status="HighsStatus.kError",
        initial_solution_acceptance="rejected",initial_solution_acceptance_evidence="controlled rejection"))
    r=execute()
    assert chosen(r,"hybrid")=={"00000","00001"}
    assert r.metadata['hybrid']['backend']['initial_solution_acceptance']=='rejected'


def test_timeout_execution_evidence_distinct_from_candidate(monkeypatch):
    monkeypatch.setattr(hybrid.importlib.util,'find_spec',lambda n: object())
    monkeypatch.setattr(process,'run_worker',lambda req,t:dict(status='external_timeout',child_created=True,
        solver_executed=True,progress=dict(initial_solution_submitted=True,initial_solution_api_status='HighsStatus.kOk',
        initial_solution_acceptance='accepted',initial_solution_acceptance_evidence='feasible MIP start')))
    r=execute(); h=r.metadata['hybrid']
    assert h['additional_solver_used'] and not h['backend']['candidate_adopted']
    assert chosen(r,'hybrid')=={'00000','00001'}


@pytest.mark.parametrize("vector,status", [([1,0,0],"completed"),([1,1,0],"completed"),
    ([1,1,1],"invalid_solution"),([.5,0,0],"invalid_solution"),([True,0,0],"invalid_response"),
    ([1,0],"invalid_response"),([float('nan'),0,0],"invalid_response")])
def test_bad_tied_lower_or_overbudget_candidate_preserves_parent(monkeypatch,vector,status):
    installed(monkeypatch,lambda req: response(req,vector=vector))
    r=execute(); b=r.metadata["hybrid"]["backend"]
    assert chosen(r,"hybrid")=={"00000","00001"} and b["status"]==status
    assert not b["candidate_adopted"]


@pytest.mark.parametrize("field,value", [("protocol_version","wrong"),("request_id","wrong"),
    ("input_fingerprint","wrong"),("variable_order_fingerprint","wrong"),("dual_bound",float('inf')),
    ("unexpected",True),("solver_executed",False),("initial_solution_acceptance","accepted"),("node_count",True)])
def test_response_contract_rejected(monkeypatch,field,value):
    installed(monkeypatch,lambda req: response(req,**{field:value}))
    r=execute()
    assert r.metadata["hybrid"]["backend"]["status"]=="invalid_response"
    assert chosen(r,"hybrid")=={"00000","00001"}


@pytest.mark.parametrize("status", ["external_timeout","child_crash","launch_error","invalid_response","unsupported_platform"])
def test_transport_failure_keeps_local_best(monkeypatch,status):
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda n: object())
    monkeypatch.setattr(process,"run_worker",lambda *a: dict(status=status,child_created=True,solver_executed=None))
    r=run(algorithm="hybrid",hybrid_config=HybridConfig(backend="highspy"))
    assert r.results["hybrid"].summary["objective_value"]==220
    assert r.metadata["hybrid"]["backend"]["status"]==status
    assert not r.results["hybrid"].summary["accepted_for_planning"]


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":',b'\xff'])
def test_strict_json(data):
    with pytest.raises((ValueError,UnicodeError)): loads(data)


def test_json_size_limit():
    from final_strategy._highspy_worker import MAX_JSON_BYTES
    with pytest.raises(ValueError,match="size"): loads(b' '* (MAX_JSON_BYTES+1))


def test_zero_backend_budget_keeps_all_scenarios(monkeypatch):
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda name: pytest.fail("no remaining time"))
    result=what_if(*population(),hybrid_config=HybridConfig(backend="highspy"),backend_time_budget=0)
    assert len(result.results)==3
    assert not result.summary.backend_child_created.any()
    assert result.summary.budget_compliant.all()


def test_whatif_spends_only_backend_allowance(monkeypatch,tmp_path):
    calls=[]
    def attempt(p,b,c,cert,tol):
        calls.append(c.external_timeout)
        return b,dict(requested=True,backend="highspy",status="external_timeout" if c.external_timeout else "skipped",
            reason=None,child_created=bool(c.external_timeout),solver_executed=False,initial_solution_submitted=False,
            initial_solution_api_status=None,initial_solution_acceptance="unknown",initial_solution_acceptance_evidence=None,
            candidate_valid=False,candidate_adopted=False,elapsed_seconds=.6 if c.external_timeout else 0.)
    monkeypatch.setattr(hybrid,"_run_backend",attempt)
    result=what_if(*population(),hybrid_config=HybridConfig(backend="highspy"),backend_time_budget=1)
    assert calls==pytest.approx([1,.4,0])
    assert result.summary.budget_compliant.all() and not result.summary.accepted_for_planning.any()
    paths=export_what_if(result,tmp_path)
    restored=json.loads(paths["metadata"].read_text(encoding='utf-8'))
    assert restored["backend_time_spent"]==1.2
    csv=pd.read_csv(paths["summary"],dtype=str,keep_default_na=False)
    assert csv.backend_status.tolist()==["external_timeout","external_timeout","skipped"]
    assert csv.solver_status.eq("").all()


@pytest.mark.parametrize("value", [-1,float('nan'),float('inf'),True])
def test_invalid_total_budget(value):
    with pytest.raises(InputValidationError):
        what_if(*population(),hybrid_config=HybridConfig(backend="highspy"),backend_time_budget=value)


def test_whatif_crash_does_not_stop_other_scenarios(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(hybrid.importlib.util,"find_spec",lambda name: object())
    def worker(req,timeout):
        calls.append(deepcopy(req))
        if len(calls)==1:
            return dict(status="child_crash",child_created=True,solver_executed=None)
        return dict(status="completed",child_created=True,solver_executed=True,
                    response=response(req),progress=None)
    monkeypatch.setattr(process,"run_worker",worker)
    result=what_if(*population(),hybrid_config=HybridConfig(backend="highspy",local_improvement=False))
    assert len(calls)==2  # 200% is already exactly certified.
    assert calls[0]['request_id']!=calls[1]['request_id']
    assert calls[0]['input_fingerprint']!=calls[1]['input_fingerprint']
    assert result.summary.backend_status.tolist()==['child_crash','completed','skipped']
    assert result.summary.objective_value.tolist()==[60,220,280]
    assert result.summary.budget_compliant.all() and not result.summary.accepted_for_planning.any()
    paths=export_what_if(result,tmp_path)
    saved=json.loads(paths['metadata'].read_text(encoding='utf-8'))
    assert saved['scenario_metadata']['budget_100']['hybrid']['selection']['source']=='highspy'


def test_backend_does_not_mutate_inputs_and_keeps_zero_cost_customers(monkeypatch):
    seg,table=population([60,100,120,5,-1],[10,20,30,0,0])
    before_seg,before_table=deepcopy(seg),table.copy(deep=True)
    seen=[]
    installed(monkeypatch,lambda req: (seen.append(deepcopy(req)) or response(req)))
    r=run(seg,table,algorithm='hybrid',hybrid_config=HybridConfig(backend='highspy',local_improvement=False))
    assert seen[0]['n']==3 and seen[0]['initial']==[1,1,0]
    assert chosen(r,'hybrid')=={'00001','00002','00003'}
    assert exact(r.metadata['hybrid']['quality']['certificate']['lower_bound'])==225
    pd.testing.assert_frame_equal(seg.customers,before_seg.customers)
    pd.testing.assert_frame_equal(seg.excluded,before_seg.excluded)
    pd.testing.assert_frame_equal(table,before_table)
    assert seg.metadata==before_seg.metadata


def test_local_termination_accounting():
    p=problem(); cert,detail=quality.improve_selection(p,["00000","00001"])
    assert cert.lower_bound==220
    assert detail["accepted_exchanges"]>=1 and detail["examined_exchanges"]>0
    assert detail["iterations"]>=1 and detail["elapsed_seconds"]>=0
    assert detail["termination_reason"] in ("no_improvement","iteration_limit","pool_exhausted","time_limit")
    _,detail=quality.improve_selection(p,["00000","00001"],time_limit=0)
    assert detail["termination_reason"]=="time_limit"


@pytest.mark.skipif(os.name!='nt',reason='Windows Job Object test')
@pytest.mark.parametrize("code,status", [("import sys;sys.stdout.write('{}')","completed"),
    ("import time;time.sleep(10)","external_timeout"),("import sys;sys.exit(3)","child_crash"),
    ("print('{')","invalid_response"),("print('{\"x\":1,\"x\":2}')","invalid_response")])
def test_real_windows_transport(code,status,tmp_path):
    result=process.run_worker(request(),.3 if status=='external_timeout' else 5,
                              command=[sys.executable,'-B','-c',code],temp_parent=tmp_path)
    assert result["status"]==status
    assert result["child_reaped"] and result["active_processes"]==0
    assert result["handles_closed"] and result["temporary_files_removed"]
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name!='nt',reason='Windows fault injection')
@pytest.mark.parametrize("method", ["CreateJobObjectW","SetInformationJobObject","CreateProcessW","AssignProcessToJobObject","ResumeThread",
                                    "WaitForSingleObject","TerminateJobObject","CloseHandle","QueryInformationJobObject"])
def test_windows_failure_cleanup(method,monkeypatch,tmp_path):
    kernel=process._kernel()
    class Fault:
        failed=False
        def __getattr__(self,name):
            real=getattr(kernel,name)
            if name!=method: return real
            def call(*args):
                if not self.failed:
                    self.failed=True
                    return 0xFFFFFFFF if name in ('ResumeThread','WaitForSingleObject') else 0
                return real(*args)
            return call
    monkeypatch.setattr(process,'_kernel',lambda: Fault())
    result=process.run_worker(request(),1,command=[sys.executable,'-B','-c','import time;time.sleep(10)'],temp_parent=tmp_path)
    assert result['status']=='launch_error'
    if result['child_created']:
        if method=='QueryInformationJobObject':
            assert result['root_reaped_after_job_close'] and not result['child_reaped']
        else:
            assert result['child_reaped']
    assert result['handles_closed'] and not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name!='nt',reason='Windows descendant test')
def test_windows_timeout_owns_descendants(tmp_path):
    code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(10)']); print(p.pid,flush=True); time.sleep(10)"
    result=process.run_worker(request(),1,command=[sys.executable,'-B','-c',code],temp_parent=tmp_path)
    assert result['status']=='external_timeout' and result['active_processes']==0 and result['child_reaped']
    assert result['temporary_files_removed']


@pytest.mark.skipif(not REAL,reason='optional highspy unavailable in this interpreter')
def test_real_highspy_adopts_and_records_acceptance():
    r=execute()
    h=r.metadata['hybrid']; b=h['backend']
    assert b['status']=='completed' and b['candidate_adopted']
    assert b['initial_solution_submitted'] and b['initial_solution_acceptance']=='accepted'
    assert 'MIP start solution is feasible' in b['initial_solution_acceptance_evidence']
    assert b['solver']['engine_version']=='1.16.0'
    assert exact(h['quality']['certificate']['lower_bound'])==220
    assert h['quality']['status']=='feasible_uncertified'


@pytest.mark.skipif(not REAL,reason='optional highspy unavailable in this interpreter')
@pytest.mark.parametrize('presolve',[True,False])
@pytest.mark.parametrize('warm',[True,False])
def test_real_worker_decimal_and_warm_cold(presolve,warm):
    req=request(); req.update(scores_hex=[float(v).hex() for v in (10,20,30)],weights=[1,1,1],capacity=1,initial=[0,0,1])
    req['options'].update(presolve=presolve,submit_initial=warm)
    transport=process.run_worker(req,10)
    assert transport['status']=='completed',transport
    r=validate_response(transport['response'],req)
    assert [round(v) for v in r['vector']]==[0,0,1]
    p=problem([10,20,30],['.050000025']*3,'.1',9)
    certificate=quality.certify_selection(p,[p.customer_ids[i] for i,v in enumerate(r['vector']) if round(v)])
    assert certificate.total_expected_cost==Decimal('.050000025')<=Decimal('.1')
    assert certificate.lower_bound==30 and certificate.optimality_proven
    assert r['initial_solution_submitted']==warm
    if not warm: assert r['initial_solution_acceptance']=='unknown'
