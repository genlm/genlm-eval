"""No-error execution backend for the runtime potential. Runs generated code
against the public test inputs and reports only whether it raised; outputs are
ignored, since a wrong answer is not an error. It mirrors the vendored run_test
(call_method for stdin, Solution()/method for functional) and runs in a forked
child, since run_test patches the interpreter and needs signal.alarm."""
from __future__ import annotations

import asyncio
import ast
import json
import math
import multiprocessing
import os
import signal
import sys
from typing import List, Optional

from genlm.eval.domains.livecodebench.vendored.testing_util import (
    Capturing,
    call_method,
    clean_if_name,
    compile_code,
    get_function,
    import_string,
    make_function,
    reliability_guard,
    timeout_handler,
    TimeoutException,
)

OK, SYNTAX, RUNTIME, TIMEOUT = "ok", "syntax", "runtime", "timeout"

_MP_CTX = None


def mp_context():
    """Context for execution children. Prefers ``forkserver`` (forks from a clean
    single-threaded server, avoiding the post-fork deadlock when the torch/vllm
    parent is multi-threaded); falls back to ``fork`` where ``__main__`` has no
    file (notebooks/-c). Override with ``LCB_MP_METHOD``."""
    global _MP_CTX
    if _MP_CTX is not None:
        return _MP_CTX
    forced = os.environ.get("LCB_MP_METHOD")
    if forced:
        _MP_CTX = multiprocessing.get_context(forced)
        return _MP_CTX
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file and os.path.exists(main_file):
        try:
            ctx = multiprocessing.get_context("forkserver")
            ctx.set_forkserver_preload(
                ["genlm.eval.domains.livecodebench.vendored.testing_util"])
            _MP_CTX = ctx
            return _MP_CTX
        except Exception:  # noqa: BLE001
            pass
    _MP_CTX = multiprocessing.get_context("fork")
    return _MP_CTX

# One semaphore per loop bounds concurrent child forks across both potentials.
_fork_sems: dict = {}


def fork_semaphore() -> "asyncio.Semaphore":
    loop = asyncio.get_running_loop()
    sem = _fork_sems.get(id(loop))
    if sem is None:
        n = int(os.environ.get("LCB_RUNTIME_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(n)
        _fork_sems[id(loop)] = sem
    return sem


def drop_trailing_compound(code: str) -> Optional[str]:
    """Drop a trailing growable compound statement (def/for/if/class/...): its
    suite can still gain lines, so a partial prefix must not execute it. Returns
    None when ``code`` does not parse (caller treats that as a syntax verdict)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    if tree.body and hasattr(tree.body[-1], "body"):
        tree.body = tree.body[:-1]
    return ast.unparse(tree)


def _classify(exc: BaseException) -> str:
    if isinstance(exc, (SyntaxError, IndentationError)):
        return SYNTAX
    # Precise type check, not a repr substring: code must not spoof TIMEOUT.
    if isinstance(exc, TimeoutException):
        return TIMEOUT
    return RUNTIME


def _run_noerror(code: str, inputs: List[str], fn_name: Optional[str],
                 drop: bool, timeout: int) -> str:
    """Execute (in the forked child) and classify the first failure."""
    reliability_guard()
    signal.signal(signal.SIGALRM, timeout_handler)
    if drop:
        code = drop_trailing_compound(code)
        if code is None:
            return SYNTAX
    if not code.strip():
        return OK  # nothing to run (e.g. only a trailing compound, now dropped)
    try:
        if fn_name:  # call-based / functional
            return _run_functional(code, inputs, fn_name, timeout)
        return _run_stdin(code, inputs, timeout)
    except BaseException as exc:  # noqa: BLE001
        signal.alarm(0)
        return _classify(exc)


def _run_functional(code: str, inputs: List[str], fn_name: str, timeout: int) -> str:
    compiled = compile_code(import_string + "\n\n" + code, timeout)
    method = get_function(compiled, fn_name) if compiled is not None else None
    if method is None:
        return OK  # entrypoint not defined (yet): nothing to run, not an error
    for inp in inputs:
        args = [json.loads(line) for line in inp.split("\n")]
        signal.alarm(timeout)
        try:
            method(*args)
        finally:
            signal.alarm(0)
    return OK


def _run_stdin(code: str, inputs: List[str], timeout: int) -> str:
    tree = ast.parse(code)  # parseable: host-gated (complete) or unparsed (prefix)
    if not any(not isinstance(s, (ast.Import, ast.ImportFrom)) for s in tree.body):
        # Only imports remain (e.g. import-only prefix): make_function would emit
        # an empty body, so just run the imports to surface any ImportError.
        signal.alarm(timeout)
        try:
            exec(compile(code, "<sol>", "exec"), {"__name__": "__main__"})
        finally:
            signal.alarm(0)
        return OK
    # Wrap like the grader (handles top-level return / __main__) and run it.
    compiled = compile_code(make_function(clean_if_name(code)), timeout)
    method = get_function(compiled, "wrapped_function") if compiled is not None else None
    if method is None:
        return OK
    for inp in inputs:
        signal.alarm(timeout)
        try:
            with Capturing():
                call_method(method, inp)
        finally:
            signal.alarm(0)
    return OK


def _child(code, inputs, fn_name, drop, timeout, conn) -> None:
    try:
        verdict = _run_noerror(code, inputs, fn_name, drop, timeout)
    except BaseException:  # noqa: BLE001
        verdict = RUNTIME
    conn.send(verdict)
    conn.close()


def run_noerror_check(code: str, inputs: List[str], fn_name: Optional[str] = None,
                      drop_trailing_compound: bool = False, timeout: float = 6.0,
                      max_total_seconds: Optional[float] = None) -> str:
    """Run ``code`` against ``inputs`` in a forked child, checking only for
    runtime/syntax errors (outputs are ignored). Returns one of
    ``OK``/``SYNTAX``/``RUNTIME``/``TIMEOUT``; ``TIMEOUT`` also covers a hung or
    crashed child (the global budget firing)."""
    run_timeout = max(1, math.ceil(timeout))  # signal.alarm needs an int; never round down
    budget = (timeout + 1) * max(1, len(inputs)) + 5  # official lcb_runner per-sample budget
    if max_total_seconds is not None:
        budget = min(budget, max_total_seconds)
    ctx = mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_child,
        args=(code, list(inputs), fn_name, drop_trailing_compound, run_timeout, child_conn),
    )
    p.start()
    child_conn.close()  # keep only the child's handle open on the write end
    verdict = TIMEOUT
    try:
        # Wait (bounded) for the child to exit before reading, so we never call
        # recv() on a live child and block forever on a partial frame.
        p.join(budget)
        if p.is_alive():
            p.kill()
            p.join()
        elif parent_conn.poll(0):
            try:
                verdict = parent_conn.recv()
            except EOFError:
                verdict = RUNTIME
        else:
            verdict = RUNTIME  # exited without sending (hard crash)
    finally:
        parent_conn.close()
    return verdict
