"""Unit tests for DS1000RuntimeNoErrorPotential wrapper logic."""

import asyncio

from genlm.eval.domains.ds1000.runtime_no_error_potential import (
    DS1000RuntimeNoErrorPotential,
)


CTX = """
def test_execution(answer):
    g = {}
    exec(answer, g, g)
    assert g.get('x') == 7
"""


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _bytes(s: str) -> list[bytes]:
    return [bytes([b]) for b in s.encode()]


def test_complete_ok_returns_zero():
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    assert _run(p.complete(_bytes("x = 7"))) == 0.0


def test_complete_bad_returns_neg_inf():
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    assert _run(p.complete(_bytes("raise RuntimeError('z')"))) == float("-inf")


def test_complete_empty_short_circuits_to_neg_inf():
    # Empty code never reaches the forked child (fast-path guard).
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    assert _run(p.complete([])) == float("-inf")
    assert p.cache_misses == 0  # didn't even hit the runner


def test_prefix_not_newline_terminated_returns_zero():
    # Newline guardrail: incomplete code lines are skipped.
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    assert _run(p.prefix(_bytes("x = 7"))) == 0.0  # no trailing \n
    assert p.cache_misses == 0


def test_cache_hit_avoids_running():
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    _run(p.complete(_bytes("x = 7")))
    _run(p.complete(_bytes("x = 7")))
    assert p.cache_misses == 1
    assert p.cache_hits == 1


def test_syntax_error_sets_flag():
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    _run(p.complete(_bytes("x = (((")))
    assert p.last_was_syntax_error is True


def test_legacy_kwargs_accepted():
    # python_executable / extra_env are no-ops but must not raise (holdout passes them).
    DS1000RuntimeNoErrorPotential(
        code_context=CTX,
        python_executable="python3",
        extra_env={"PYTHONHASHSEED": "0"},
    )


def test_coerce_inherits_config():
    """coerce() returns a fresh instance with the same context + timeout."""
    class _Other:
        vocab = [bytes([i]) for i in range(256)]
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=7.5)
    q = p.coerce(_Other())
    assert q.code_context == CTX
    assert q.timeout_seconds == 7.5
    assert q is not p


def test_prefix_with_newline_invokes_scorer():
    """A newline-terminated prefix runs the full scoring path (not the guardrail)."""
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    assert _run(p.prefix(_bytes("x = 7\n"))) == 0.0
    assert p.cache_misses == 1


def test_bytes_to_str_input_shapes():
    """Cover all input-type branches: str, bytes, list[int], list[bytes], list[other]."""
    p = DS1000RuntimeNoErrorPotential(code_context=CTX)
    assert p._bytes_to_str("abc") == "abc"
    assert p._bytes_to_str(b"abc") == "abc"
    assert p._bytes_to_str([ord("a"), ord("b")]) == "ab"
    assert p._bytes_to_str([b"a", b"b"]) == "ab"
    assert p._bytes_to_str([]) == ""
    assert p._bytes_to_str(["a", "b"]) == "ab"  # generic str fallback


def test_f_transform_is_applied():
    """If ``f`` is given, both prefix() and complete() route context through it."""
    seen = []
    def my_f(ctx):
        seen.append("called")
        return ctx
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0, f=my_f)
    _run(p.complete(_bytes("x = 7")))
    _run(p.prefix(_bytes("x = 7\n")))
    assert len(seen) == 2


def test_cache_evicts_oldest_when_full():
    """LRU cache must evict when it exceeds ``_score_cache_maxsize``."""
    p = DS1000RuntimeNoErrorPotential(code_context=CTX, timeout_seconds=10.0)
    p._score_cache_maxsize = 2
    _run(p.complete(_bytes("x = 7")))
    _run(p.complete(_bytes("x = 8")))
    _run(p.complete(_bytes("x = 9")))
    assert len(p._score_cache) == 2
