"""Subprocess wrapper around the vendored LCB ``run_test``: it patches the
interpreter destructively (``reliability_guard``) and needs main-thread
``signal.alarm``, so it must run in a forked child, as in official lcb_runner."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from genlm.eval.domains.livecodebench import capture
from genlm.eval.domains.livecodebench.runtime_execution import mp_context
from genlm.eval.domains.livecodebench.vendored.testing_util import run_test


def _child_run(sample, generation, debug, conn, timeout, capture_enabled):
    # A forkserver child forks from the clean server, not the parent, so it does
    # not inherit capture's grade-function patch; enable it here when the parent did.
    if capture_enabled:
        capture.enable_capture()
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    conn.send((res, metadata))
    conn.close()


def check_correctness(sample: Dict[str, str], generation: str,
                      timeout: float = 6.0, debug: bool = False,
                      max_total_seconds: Optional[float] = None,
                      ) -> Tuple[List[Any], Dict[str, Any]]:
    """Run ``generation`` against the tests in a forked child.

    ``results`` is per-test ``True``/``False`` (or sentinel ints ``-1``/``-2``/``-4``
    on failure). ``max_total_seconds`` caps the official per-sample wall-clock budget
    of ``(timeout + 1) * n_tests + 5`` — the budget only binds when generated code
    hangs in a way ``signal.alarm`` can't interrupt, so capping it bounds the stall
    from a single pathological generation without affecting normal grading."""
    # Guard the input_output parse (official lcb_runner does this unguarded, but our
    # from_jsonl allows prompts-only snapshots where it may be absent/malformed): a bad
    # sample scores fail instead of crashing the whole eval run.
    try:
        n_tests = len(json.loads(sample["input_output"])["inputs"])
    except (KeyError, TypeError, ValueError):
        return [-1], {"error": "missing or malformed eval_sample"}
    run_timeout = max(1, math.ceil(timeout))  # signal.alarm needs an int; never round down
    budget = (timeout + 1) * n_tests + 5  # official lcb_runner per-sample budget
    if max_total_seconds is not None:
        budget = min(budget, max_total_seconds)
    ctx = mp_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_child_run,
        args=(sample, generation, debug, child_conn, run_timeout, capture.is_enabled()),
    )
    p.start()
    child_conn.close()  # keep only the child's handle open on the write end
    try:
        # Wait (bounded) for the child to exit before reading, so we never call
        # recv() on a live child and block forever on a partial frame.
        p.join(budget)
        if p.is_alive():
            p.kill()
            p.join()
        elif parent_conn.poll(0):
            try:
                res, metadata = parent_conn.recv()
                return list(res), dict(metadata)
            except EOFError:  # crashed mid-send
                pass
    finally:
        parent_conn.close()
    return [-1] * n_tests, {"error": "global timeout or crashed child"}


def passed_all(sample: Dict[str, str], generation: str, timeout: float = 6.0,
               max_total_seconds: Optional[float] = None) -> bool:
    """True iff every test passed (``> 0``), matching official ``np.all(gen > 0)``."""
    results, _ = check_correctness(sample, generation, timeout=timeout,
                                   max_total_seconds=max_total_seconds)
    return bool(results) and all(r > 0 for r in results)
