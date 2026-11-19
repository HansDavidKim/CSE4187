"""Private v1 JSON worker. Importing this module never imports highspy.

Native solver output is redirected to stderr; stdout contains one JSON response.
The parent validates the protocol and independently certifies the returned IDs.
"""
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

VERSION = "kit-highspy-v1"
MAX_JSON_BYTES = 16 * 1024 * 1024
IDENTITY = ("protocol_version", "request_id", "input_fingerprint", "variable_order_fingerprint")
REQUEST_FIELDS = set(IDENTITY) | {"n", "customer_ids", "scores_hex", "weights", "capacity", "initial", "options"}
RESPONSE_FIELDS = set(IDENTITY) | {"solver_status", "solver_message", "vector", "primal_objective",
    "dual_bound", "mip_gap", "node_count", "solver_executed", "initial_solution_submitted",
    "initial_solution_api_status", "initial_solution_acceptance", "initial_solution_acceptance_evidence",
    "timings", "engine_version"}


def loads(data):
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON size limit exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _number(value, *, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def validate_request(request):
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise ValueError("invalid request fields")
    if request["protocol_version"] != VERSION or any(not isinstance(request[k], str) or not request[k] for k in IDENTITY):
        raise ValueError("invalid request identity")
    n = request["n"]
    if type(n) is not int or not 1 <= n <= 1000000:
        raise ValueError("invalid variable count")
    for key in ("customer_ids", "scores_hex", "weights", "initial"):
        if not isinstance(request[key], list) or len(request[key]) != n:
            raise ValueError("request vector length mismatch")
    if (any(not isinstance(v, str) for v in request["customer_ids"])
            or len(set(request["customer_ids"])) != n):
        raise ValueError("invalid ordered customer IDs")
    scores = [float.fromhex(v) for v in request["scores_hex"]]
    if any(not math.isfinite(v) or v <= 0 for v in scores):
        raise ValueError("invalid binary64 score")
    if any(type(v) is not int or not 0 < v <= 10**12 for v in request["weights"]):
        raise ValueError("invalid integer weights")
    if type(request["capacity"]) is not int or not 0 <= request["capacity"] <= 2**53-1 or sum(request["weights"]) > 2**53-1:
        raise ValueError("unsafe integer scale")
    if any(type(v) is not int or v not in (0, 1) for v in request["initial"]):
        raise ValueError("invalid initial binary vector")
    if sum(w*x for w, x in zip(request["weights"], request["initial"])) > request["capacity"]:
        raise ValueError("infeasible initial solution")
    options = request["options"]
    if not isinstance(options, dict) or set(options) != {"time_limit", "presolve", "threads", "mip_rel_gap", "submit_initial"}:
        raise ValueError("invalid solver options")
    if (not _number(options["time_limit"], nonnegative=True) or not 0 <= options["time_limit"] <= 3600
            or not _number(options["mip_rel_gap"], nonnegative=True) or options["mip_rel_gap"] > 1
            or type(options["threads"]) is not int or not 1 <= options["threads"] <= 64
            or type(options["presolve"]) is not bool or type(options["submit_initial"]) is not bool):
        raise ValueError("invalid solver option value")
    return scores


def validate_response(response, request):
    if not isinstance(response, dict) or set(response) != RESPONSE_FIELDS:
        raise ValueError("unexpected response fields")
    if any(response[k] != request[k] for k in IDENTITY):
        raise ValueError("response identity/fingerprint mismatch")
    for key in ("solver_status", "solver_message", "engine_version"):
        if not isinstance(response[key], str):
            raise ValueError("invalid response status/version")
    for key in ("solver_executed", "initial_solution_submitted"):
        if type(response[key]) is not bool:
            raise ValueError("invalid execution evidence")
    if response["initial_solution_acceptance"] not in ("accepted", "rejected", "unknown"):
        raise ValueError("invalid acceptance state")
    for key in ("initial_solution_api_status", "initial_solution_acceptance_evidence"):
        if response[key] is not None and not isinstance(response[key], str):
            raise ValueError("invalid initial-solution evidence")
    if (response["initial_solution_acceptance"] != "unknown"
            and (not response["initial_solution_submitted"] or not response["initial_solution_acceptance_evidence"])):
        raise ValueError("initial solution acceptance has no supporting evidence")
    for key in ("primal_objective", "dual_bound", "mip_gap"):
        if response[key] is not None and not _number(response[key], nonnegative=key == "mip_gap"):
            raise ValueError("nonfinite/invalid solver diagnostic")
    if response["node_count"] is not None and (type(response["node_count"]) is not int or response["node_count"] < 0):
        raise ValueError("invalid node count")
    if not isinstance(response["timings"], dict) or set(response["timings"]) != {"model_seconds", "initial_solution_seconds", "solve_seconds", "worker_seconds"}:
        raise ValueError("invalid worker timings")
    if any(not _number(v, nonnegative=True) for v in response["timings"].values()):
        raise ValueError("invalid worker time")
    vector = response["vector"]
    if vector is not None and not response["solver_executed"]:
        raise ValueError("candidate without solver execution")
    if vector is not None and (not isinstance(vector, list) or len(vector) != request["n"]
                               or any(not _number(v) for v in vector)):
        raise ValueError("invalid solution vector shape/numbers")
    return response


def solve(request, progress_path=None):
    started = perf_counter()
    scores = validate_request(request)
    import numpy as np
    import highspy
    h = highspy.Highs()
    options = request["options"]
    for key, value in (("presolve", "on" if options["presolve"] else "off"),
                       ("time_limit", options["time_limit"]), ("mip_rel_gap", options["mip_rel_gap"]),
                       ("threads", options["threads"]), ("parallel", "on" if options["threads"] > 1 else "off")):
        if h.setOptionValue(key, value) != highspy.HighsStatus.kOk:
            raise RuntimeError("highspy option rejected: " + key)
    n, scale = request["n"], max(scores)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, 1
    lp.col_cost_ = -np.asarray(scores, dtype=float) / scale
    lp.col_lower_, lp.col_upper_ = np.zeros(n), np.ones(n)
    lp.row_lower_, lp.row_upper_ = np.array([-highspy.kHighsInf]), np.array([request["capacity"]], dtype=float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_ = np.arange(n+1, dtype=np.int32), np.zeros(n, dtype=np.int32)
    lp.a_matrix_.value_ = np.array(request["weights"], dtype=float)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    if h.passModel(lp) != highspy.HighsStatus.kOk:
        raise RuntimeError("highspy model rejected")
    result = {k: request[k] for k in IDENTITY}
    result.update(initial_solution_submitted=False, initial_solution_api_status=None,
                  initial_solution_acceptance="unknown", initial_solution_acceptance_evidence=None,
                  solver_executed=False)
    def progress():
        if progress_path is not None:
            # This request-owned journal is evidence only, never a candidate solution.
            Path(progress_path).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    running = False
    def logging(event):
        if running and not result["solver_executed"]:
            result["solver_executed"] = True
            progress()
        message = event.message.strip()
        if "MIP start solution is feasible" in message:
            result.update(initial_solution_acceptance="accepted", initial_solution_acceptance_evidence=message)
            progress()
        elif "MIP start solution is infeasible" in message:
            result.update(initial_solution_acceptance="rejected", initial_solution_acceptance_evidence=message)
            progress()
    h.cbLogging.subscribe(logging)
    model_time = perf_counter()-started
    tick = perf_counter()
    if options["submit_initial"]:
        status = h.setSolution(n, np.arange(n, dtype=np.int32), np.array(request["initial"], dtype=float))
        result.update(initial_solution_submitted=True, initial_solution_api_status=str(status))
    initial_time = perf_counter()-tick
    progress()
    tick = perf_counter()
    running = True
    run_status = h.run()
    result["solver_executed"] = True
    solve_time = perf_counter()-tick
    info, solution = h.getInfo(), h.getSolution()
    def finite(value):
        return float(value) if math.isfinite(value) else None
    result.update(solver_status=h.modelStatusToString(h.getModelStatus()), solver_message=str(run_status),
        vector=list(solution.col_value) if solution.value_valid else None,
        # Diagnostics remain in the native normalized minimization objective.
        primal_objective=finite(info.objective_function_value), dual_bound=finite(info.mip_dual_bound),
        mip_gap=finite(info.mip_gap), node_count=max(0, int(info.mip_node_count)), engine_version=h.version(),
        timings={"model_seconds": model_time, "initial_solution_seconds": initial_time,
                 "solve_seconds": solve_time, "worker_seconds": perf_counter()-started})
    return result


if __name__ == "__main__":
    # Redirect the OS descriptor too: native C++ output must not corrupt JSON.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    request = loads(sys.stdin.buffer.read(MAX_JSON_BYTES+1))
    response = solve(request, sys.argv[1])
    protocol.write(json.dumps(response, allow_nan=False))
    protocol.flush()
