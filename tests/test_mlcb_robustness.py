"""Robustness and resource-handling tests for the multilingual-LCB executor.

These use only python (no external toolchain) so they run in every lane, and cover the failure
modes a grader must survive: runaway programs, flooded stdout, leaked child processes, the
per-language memory cap, the prepare() hooks, and concurrent grading. Behaviors here were
confirmed against the live executor before being pinned.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from genlm.eval.domains.livecodebench_multilingual import LocalSubprocessExecutor
from genlm.eval.domains.livecodebench_multilingual import executor as executor_mod
from genlm.eval.domains.livecodebench_multilingual.vendored.testing_plang import (
    SubprocessConfig,
)


def _pgrep(pattern):
    """PIDs whose command line matches `pattern` (empty list if none)."""
    found = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [p for p in found.stdout.split() if p]


def test_timeout_is_bounded_and_classified():
    # An infinite loop must be killed near the timeout, not run forever, and be reported as a
    # timeout (not a wrong answer): status TimeoutExpired, the test scored EXECFAIL, not solved.
    ex = LocalSubprocessExecutor()
    t0 = time.time()
    solved, meta = ex.run("while True:\n    pass\n", ["x\n"], ["x\n"], "python", 1.0)
    elapsed = time.time() - t0
    assert solved is False
    assert meta["status"] == "TimeoutExpired"
    assert meta["per_test"] == ["EXECFAIL"]
    assert elapsed < 15.0, f"timeout took {elapsed:.1f}s; group kill may be leaking"


def test_large_output_does_not_hang():
    # A program flooding stdout (~8 MB) must be graded (here: wrong answer) within a bounded
    # time, never deadlocking on a full pipe buffer.
    ex = LocalSubprocessExecutor()
    flood = "import sys\nsys.stdout.write('z' * 8_000_000)\n"
    t0 = time.time()
    solved, meta = ex.run(flood, ["3\n1 2 3\n"], ["6\n"], "python", 5.0)
    elapsed = time.time() - t0
    assert solved is False
    assert meta["per_test"] == ["FAILED"]
    assert elapsed < 15.0, f"large-output grading took {elapsed:.1f}s"


def test_grandchild_reaped_after_timeout():
    # A timed-out program that spawned its own child must not leak it: the executor runs each
    # candidate in a fresh session and SIGKILLs the whole process group, so the grandchild dies.
    sentinel = "mlcb_robustness_sentinel_b91e"
    prog = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', '{sentinel}'])\n"
        "while True:\n    time.sleep(0.05)\n"
    )
    ex = LocalSubprocessExecutor()
    solved, _ = ex.run(prog, ["x\n"], ["x\n"], "python", 1.0)
    assert solved is False
    # Poll for the group kill to propagate (a fixed sleep can flake on a loaded node).
    survivors = _pgrep(sentinel)
    deadline = time.time() + 5.0
    while survivors and time.time() < deadline:
        time.sleep(0.1)
        survivors = _pgrep(sentinel)
    for pid in survivors:  # never leave a leak behind even if the assert fails
        subprocess.run(["kill", "-9", pid], capture_output=True)
    assert survivors == [], f"leaked child processes: {survivors}"


@pytest.mark.parametrize(
    "language,capped",
    [
        ("python", True),
        ("c++", True),
        ("go", True),
        ("java", True),
        ("rust", False),
        ("javascript", False),
        ("typescript", False),
        ("julia", False),
    ],
)
def test_memory_limit_config(language, capped):
    # Native/GC runtimes (rust, the JS engines, julia) reserve huge virtual address space at
    # startup, so an RLIMIT_AS cap spuriously kills them; they are exempt. Interpreted/JVM
    # languages keep the cap. This pins that policy without allocating any memory.
    assert SubprocessConfig(plang=language).limit_memory is capped


def test_prepare_julia_warms_once(monkeypatch):
    # Julia's first cold run builds its precompile cache and can exceed a per-test timeout, so
    # prepare() warms it once; a second prepare() for the same language is a no-op.
    calls = []
    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: True)
    monkeypatch.setattr(
        executor_mod.subprocess, "run", lambda *a, **k: calls.append(a[0])
    )
    ex = LocalSubprocessExecutor()
    ex.prepare("julia")
    ex.prepare("julia")
    assert len(calls) == 1, f"expected one warmup call, got {calls}"
    assert "julia" in calls[0][0]


@pytest.mark.parametrize("language", ["go", "python"])
def test_prepare_makes_no_subprocess_call(monkeypatch, language):
    # go must not clear its build cache in prepare() (a cold stdlib rebuild blows the build
    # timeout); python needs no setup. Neither should shell out.
    calls = []
    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: True)
    monkeypatch.setattr(executor_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    LocalSubprocessExecutor().prepare(language)
    assert calls == []


def test_prepare_missing_toolchain_raises_and_not_marked(monkeypatch):
    # A missing toolchain must fail loudly and must not be recorded as prepared (so a later
    # install + retry works without a stale cache entry).
    monkeypatch.setattr(executor_mod, "is_toolchain_available", lambda lang: False)
    ex = LocalSubprocessExecutor()
    with pytest.raises(RuntimeError, match="toolchain"):
        ex.prepare("c++")
    assert "c++" not in ex._prepared


def test_unicode_io_round_trips():
    # Non-ASCII stdin/stdout must survive the write-pipe-read-compare path (UTF-8), so a program
    # that echoes its input matches an identical expected output and mismatches a different one.
    ex = LocalSubprocessExecutor()
    echo = "import sys\nsys.stdout.write(sys.stdin.read())\n"
    payload = ["héllo wörld ☃\n"]
    solved, meta = ex.run(echo, payload, payload, "python", 5.0)
    assert solved is True and meta["per_test"] == ["PASSED"]
    bad, _ = ex.run(echo, payload, ["different\n"], "python", 5.0)
    assert bad is False


def test_concurrent_grading_no_crosstalk():
    # Many candidates graded at once must not collide on temp files or stdin: each verdict must
    # match its own program. Mixes correct and wrong solutions across threads.
    correct = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()[1:]))\n"
    wrong = "import sys\nprint(sum(int(x) for x in sys.stdin.read().split()[1:]) + 1)\n"
    jobs = [(correct, True), (wrong, False)] * 6

    def grade(job):
        code, _ = job
        ex = LocalSubprocessExecutor()
        solved, _ = ex.run(code, ["3\n1 2 3\n"], ["6\n"], "python", 10.0)
        return solved

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(grade, jobs))
    assert results == [expected for _, expected in jobs]
