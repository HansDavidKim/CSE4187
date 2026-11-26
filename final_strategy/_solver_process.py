"""Private Windows Job Object runner; only request-owned processes/files are managed."""
import ctypes as c
from ctypes import wintypes as w
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter, sleep

from ._highspy_worker import MAX_JSON_BYTES, IDENTITY, loads


class BasicLimits(c.Structure):
    _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong), ("flags", w.DWORD),
                ("min_working", c.c_size_t), ("max_working", c.c_size_t), ("active_limit", w.DWORD),
                ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class ExtendedLimits(c.Structure):
    _fields_ = [("basic", BasicLimits), ("io", c.c_ulonglong*6), ("process_memory", c.c_size_t),
                ("job_memory", c.c_size_t), ("peak_process_memory", c.c_size_t), ("peak_job_memory", c.c_size_t)]


class Accounting(c.Structure):
    _fields_ = [("times", c.c_longlong*4), ("faults", w.DWORD), ("total", w.DWORD),
                ("active", w.DWORD), ("terminated", w.DWORD)]


class Startup(c.Structure):
    _fields_ = [("cb", w.DWORD), ("reserved", w.LPWSTR), ("desktop", w.LPWSTR), ("title", w.LPWSTR),
                *[(name, w.DWORD) for name in ("x", "y", "xs", "ys", "xc", "yc", "fill", "flags")],
                ("show", w.WORD), ("reserved_size", w.WORD), ("reserved_bytes", c.POINTER(c.c_byte)),
                ("stdin", w.HANDLE), ("stdout", w.HANDLE), ("stderr", w.HANDLE)]


class StartupEx(c.Structure):
    _fields_ = [("startup", Startup), ("attributes", c.c_void_p)]


class ProcessInfo(c.Structure):
    _fields_ = [("process", w.HANDLE), ("thread", w.HANDLE), ("pid", w.DWORD), ("tid", w.DWORD)]


def _kernel():
    k = c.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
        "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
        "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, c.c_void_p, c.c_void_p, w.BOOL, w.DWORD,
                            c.c_void_p, w.LPCWSTR, c.c_void_p, c.POINTER(ProcessInfo)], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "ResumeThread": ([w.HANDLE], w.DWORD), "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL), "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
        "GetExitCodeProcess": ([w.HANDLE, c.POINTER(w.DWORD)], w.BOOL), "CloseHandle": ([w.HANDLE], w.BOOL),
        "InitializeProcThreadAttributeList": ([c.c_void_p, w.DWORD, w.DWORD, c.POINTER(c.c_size_t)], w.BOOL),
        "UpdateProcThreadAttribute": ([c.c_void_p, w.DWORD, c.c_size_t, c.c_void_p, c.c_size_t,
                                       c.c_void_p, c.c_void_p], w.BOOL),
        "DeleteProcThreadAttributeList": ([c.c_void_p], None),
    }
    for name, (arguments, result) in signatures.items():
        method = getattr(k, name)
        method.argtypes, method.restype = arguments, result
    return k


def _check(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


def _run_owned(command, stdin, stdout, stderr, timeout, result):
    """Failure paths retain process ownership until cleanup; result records cleanup evidence."""
    import msvcrt
    k = _kernel()
    job, pi, assigned, attributes = None, ProcessInfo(), False, None
    limits, accounting = ExtendedLimits(), Accounting()
    started = perf_counter()
    try:
        job = _check(k.CreateJobObjectW(None, None))
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no memory limit.
        _check(k.SetInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits)))
        size = c.c_size_t()
        k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        storage = c.create_string_buffer(size.value)
        _check(k.InitializeProcThreadAttributeList(storage, 1, 0, c.byref(size)))
        attributes = storage
        handles = [msvcrt.get_osfhandle(stream.fileno()) for stream in (stdin, stdout, stderr)]
        inherited = (w.HANDLE*3)(*handles)
        _check(k.UpdateProcThreadAttribute(storage, 0, 0x20002, inherited, c.sizeof(inherited), None, None))
        si = StartupEx()
        si.startup.cb, si.startup.flags = c.sizeof(si), 0x101
        si.startup.stdin, si.startup.stdout, si.startup.stderr = handles
        si.attributes = c.cast(storage, c.c_void_p)
        try:
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            command_line = c.create_unicode_buffer(subprocess.list2cmdline(command))
            _check(k.CreateProcessW(None, command_line, None, None, True, 0x08080004,
                                   None, None, c.byref(si), c.byref(pi)))
            result.update(child_created=True, pid=int(pi.pid), worker_creation_seconds=perf_counter()-started)
        finally:
            for handle in handles:
                os.set_handle_inheritable(handle, False)
        _check(k.AssignProcessToJobObject(job, pi.process))
        assigned = True
        if k.ResumeThread(pi.thread) == 0xFFFFFFFF:
            raise c.WinError(c.get_last_error())
        result["child_resumed"] = True
        deadline = started+timeout
        while True:
            wait = k.WaitForSingleObject(pi.process, 20)
            if wait not in (0, 258):
                raise c.WinError(c.get_last_error())
            if os.fstat(stdout.fileno()).st_size > MAX_JSON_BYTES or os.fstat(stderr.fileno()).st_size > MAX_JSON_BYTES:
                result.update(status="invalid_response", reason="worker output size limit")
                break
            if wait == 0:
                code = w.DWORD()
                _check(k.GetExitCodeProcess(pi.process, c.byref(code)))
                result.update(returncode=int(code.value), status="completed" if code.value == 0 else "child_crash")
                break
            if perf_counter() >= deadline:
                result.update(status="external_timeout", external_timeout=True)
                break
    except Exception as exc:
        result.update(status="launch_error", reason=f"{type(exc).__name__}: {exc}")
    finally:
        cleanup_errors = []
        result["handles_closed"] = True
        def close_owned(handle, label):
            if handle and not k.CloseHandle(handle):
                cleanup_errors.append(f"CloseHandle({label}) failed; retried")
                if not k.CloseHandle(handle):
                    result["handles_closed"] = False
                    return False
            return True
        if pi.process:
            try:
                if assigned:
                    if not k.TerminateJobObject(job, 1):
                        cleanup_errors.append("TerminateJobObject failed; retried")
                        _check(k.TerminateJobObject(job, 1))
                else:
                    _check(k.TerminateProcess(pi.process, 1))
                if k.WaitForSingleObject(pi.process, 5000) != 0:
                    raise RuntimeError("owned root termination not confirmed")
                if assigned:
                    deadline = perf_counter()+5
                    while True:
                        _check(k.QueryInformationJobObject(job, 1, c.byref(accounting), c.sizeof(accounting), None))
                        if not accounting.active:
                            result.update(child_reaped=True, active_processes=0)
                            break
                        if perf_counter() >= deadline:
                            raise RuntimeError("owned descendants termination not confirmed")
                        sleep(.01)
                    if k.QueryInformationJobObject(job, 9, c.byref(limits), c.sizeof(limits), None):
                        result["peak_job_commit_bytes"] = int(limits.peak_job_memory)
                else:
                    # Suspended root never ran and therefore has no children.
                    result.update(child_reaped=True, active_processes=0)
            except Exception as exc:
                cleanup_errors.append(str(exc))
                # Closing the owned kill-on-close Job is the last-resort cleanup.
                if not assigned:
                    k.TerminateProcess(pi.process, 1)
                    k.WaitForSingleObject(pi.process, 5000)
        if attributes is not None:
            k.DeleteProcThreadAttributeList(attributes)
        if job:
            if close_owned(job, "job") and pi.process and not result.get("child_reaped"):
                # Kill-on-close fallback: wait for our root, but do not claim that
                # all descendants were observed dead without Job accounting.
                result["root_reaped_after_job_close"] = k.WaitForSingleObject(pi.process, 5000) == 0
        for handle in (pi.thread, pi.process):
            close_owned(handle, "process/thread")
        if cleanup_errors:
            result.update(status="launch_error", cleanup_errors=cleanup_errors)
        result["process_seconds"] = perf_counter()-started
        if not result["child_resumed"]:
            result["solver_executed"] = False


def run_worker(request, timeout, *, command=None, temp_parent=None):
    """Return transport evidence, never a trusted allocation. Optional args are test-only internals."""
    result = {"status": "launch_error", "child_created": False, "child_resumed": False,
              "child_reaped": False, "external_timeout": False, "solver_executed": None,
              "response": None, "progress": None, "returncode": None}
    if os.name != "nt":
        return {**result, "status": "unsupported_platform"}
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        return {**result, "reason": "positive finite external timeout required"}
    start = perf_counter()
    try:
        content = json.dumps(request, allow_nan=False).encode("utf-8")
        if len(content) > MAX_JSON_BYTES:
            return {**result, "status": "invalid_response", "reason": "request size limit"}
        with tempfile.TemporaryDirectory(prefix="kit-highspy-", dir=temp_parent) as directory:
            root = Path(directory)
            request_path, output_path, log_path, progress_path = [root/name for name in ("request.json", "response.json", "stderr.log", "progress.json")]
            request_path.write_bytes(content)
            result["model_transfer_seconds"] = perf_counter()-start
            argv = command if command is not None else [sys.executable, "-B", str(Path(__file__).with_name("_highspy_worker.py")), str(progress_path)]
            with request_path.open("rb") as input_stream, output_path.open("w+b") as output, log_path.open("w+b") as log:
                _run_owned(argv, input_stream, output, log, timeout, result)
            tick = perf_counter()
            if progress_path.exists() and progress_path.stat().st_size <= 16384:
                try:
                    progress = loads(progress_path.read_bytes())
                    allowed = set(IDENTITY) | {"solver_executed", "initial_solution_submitted", "initial_solution_api_status",
                                              "initial_solution_acceptance", "initial_solution_acceptance_evidence"}
                    if (isinstance(progress, dict) and set(progress) == allowed
                            and all(progress[k] == request[k] for k in IDENTITY)
                            and type(progress["solver_executed"]) is bool
                            and type(progress["initial_solution_submitted"]) is bool
                            and progress["initial_solution_acceptance"] in ("accepted", "rejected", "unknown")
                            and (progress["initial_solution_acceptance"] == "unknown" or
                                 (progress["initial_solution_submitted"] and progress["initial_solution_acceptance_evidence"]))
                            and all(progress[k] is None or isinstance(progress[k], str) for k in
                                    ("initial_solution_api_status", "initial_solution_acceptance_evidence"))):
                        result.update(progress=progress, solver_executed=True if progress["solver_executed"] else None)
                except (ValueError, KeyError):
                    pass  # A partially written journal cannot certify execution or a solution.
            if result["status"] == "completed":
                try:
                    with output_path.open("rb") as stream:
                        result["response"] = loads(stream.read(MAX_JSON_BYTES+1))
                except (ValueError, UnicodeError) as exc:
                    result.update(status="invalid_response", reason=str(exc))
            with log_path.open("rb") as stream:
                stream.seek(max(0, log_path.stat().st_size-8192))
                result["stderr_tail"] = stream.read(8192).decode("utf-8", errors="replace")
            result["result_receive_seconds"] = perf_counter()-tick
        result["temporary_files_removed"] = True
    except Exception as exc:
        result.update(status="launch_error", reason=f"{type(exc).__name__}: {exc}")
    result["external_seconds"] = perf_counter()-start
    return result
