"""Unit tests for DS-1000's fork-per-request critic backend."""

import asyncio
import time

from genlm.eval.domains.ds1000.runtime_no_error_potential import _fork_score


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


CTX = """
def test_execution(answer):
    g = {}
    exec(answer, g, g)
    assert g.get('x') == 7
"""

CTX_NO_TE = "y = 1"

CTX_TE_RAISES = """
def test_execution(answer):
    raise ValueError("boom")
"""


def test_ok():
    assert _run(_fork_score(CTX, "x = 7", 10.0)) == "OK"


def test_assertion_error_counts_as_ok():
    # te()'s assert fails (x=8); DS-1000 convention treats this as OK.
    assert _run(_fork_score(CTX, "x = 8", 10.0)) == "OK"


def test_runtime_error_returns_bad():
    assert _run(_fork_score(CTX, "raise RuntimeError('z')", 10.0)) == "BAD"


def test_syntax_error_returns_syn():
    assert _run(_fork_score(CTX, "x = (((", 10.0)) == "SYN"


def test_missing_test_execution_returns_bad():
    assert _run(_fork_score(CTX_NO_TE, "x = 7", 10.0)) == "BAD"


def test_te_non_assertion_exception_returns_bad():
    assert _run(_fork_score(CTX_TE_RAISES, "x = 7", 10.0)) == "BAD"


def test_timeout_returns_none_and_kills_child():
    t0 = time.perf_counter()
    result = _run(_fork_score(CTX, "while True: pass", 0.5))
    assert result is None
    assert time.perf_counter() - t0 < 5.0  # killed promptly, didn't hang


def test_state_isolation_between_calls():
    """Call K's module-level mutations must not leak into call K+1."""
    # seed(0)'s first np.random.random() is ~0.5488; if it leaked, call 2 would see it.
    ctx = """
import numpy as np
def test_execution(answer):
    g = {}
    exec(answer, g, g)
    assert abs(float(np.random.random()) - 0.5488135) > 1e-6
"""
    assert _run(_fork_score(ctx, "import numpy as np; np.random.seed(0)", 10.0)) == "OK"
    assert _run(_fork_score(ctx, "pass", 10.0)) == "OK"


def test_child_stdout_does_not_leak(capfd):
    """A candidate's prints must not leak into the parent's stdout/stderr."""
    _run(_fork_score(CTX, "print('HELLO_FROM_CHILD'); x = 7", 10.0))
    out, err = capfd.readouterr()
    assert "HELLO_FROM_CHILD" not in out
    assert "HELLO_FROM_CHILD" not in err


def test_concurrent_dispatch():
    async def go():
        return await asyncio.gather(
            _fork_score(CTX, "x = 7", 10.0),
            _fork_score(CTX, "x = 8", 10.0),
            _fork_score(CTX, "x = (((", 10.0),
            _fork_score(CTX, "raise RuntimeError('z')", 10.0),
        )
    assert _run(go()) == ["OK", "OK", "SYN", "BAD"]
