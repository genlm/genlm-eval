"""MBPP code execution and outcome classification.

Runs a candidate solution against an MBPP problem's ``assert`` tests in a sandboxed
subprocess and returns a structured result. Shared by the evaluator and both potentials.
Deliberately free of ``genlm.control`` / ML imports so it stays cheap to import and easy to
unit-test.

MBPP tests are self-contained ``assert`` statements that call the solution's function by its
real name (e.g. ``assert similar_elements((3,4),(4,5)) == (4,)``), so grading is just:
``exec(test_setup_code); exec(code); exec(test)`` per test, in a fresh namespace.

Two notions of outcome, matching the two potentials:
  * runtime-no-error: the code loads and every test runs without a NON-assertion exception
    (a failed ``assert`` = wrong answer = still "ran"); a syntax error or any other raised
    exception = not no-error.
  * test-passing: how many tests pass outright (no exception at all).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional

# Bound concurrent solution subprocesses per event loop (SMC runs many particles at once),
# so a burst of forks does not exhaust the OS thread/pid limits. Sized from the env.
_subprocess_sems: dict = {}


def subprocess_semaphore() -> "asyncio.Semaphore":
    loop = asyncio.get_running_loop()
    sem = _subprocess_sems.get(id(loop))
    if sem is None:
        n = int(os.environ.get("MBPP_SUBPROCESS_CONCURRENCY", "8"))
        sem = asyncio.Semaphore(n)
        _subprocess_sems[id(loop)] = sem
    return sem

# Per-test outcomes reported by the harness.
_PASS = "pass"          # assert held
_ASSERT = "assert"      # AssertionError -> ran fine, wrong answer
# anything else is the exception type name (e.g. "NameError", "TypeError")

_MARKER = "__MBPP_RESULT__"

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OPEN_FENCE = re.compile(r"```(?:python|py)?\s*\n", re.IGNORECASE)


def extract_code(text: str) -> str:
    """Extract the Python solution from a full model output.

    Returns the last fenced ```python block if present, otherwise the stripped text.
    Reasoning traces (``<think>...</think>``) are dropped so code fenced inside the
    reasoning is never mistaken for the answer.
    """
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def extract_code_prefix(text: str) -> str:
    """Extract code from a partial generation (a prefix).

    Like :func:`extract_code`, but if the last fence is still open (no closing ```), take
    everything after it -- the solution is mid-stream.
    """
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    opens = list(_OPEN_FENCE.finditer(text))
    if opens:
        return text[opens[-1].end():]
    return text.strip()


@dataclass
class MBPPRunResult:
    """Structured outcome of running a candidate against an MBPP problem's tests."""

    syntax_error: bool = False          # code did not parse
    load_error: Optional[str] = None    # exception type raised while exec-ing setup/code
    per_test: List[str] = field(default_factory=list)  # _PASS / _ASSERT / "<ExcName>"
    n_tests: int = 0
    timeout: bool = False               # killed by the wall-clock limit
    crashed: bool = False               # no verdict marker (segfault / hard crash)

    @property
    def loaded(self) -> bool:
        return not self.syntax_error and self.load_error is None and not self.timeout and not self.crashed

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.per_test if r == _PASS)

    @property
    def n_runtime_errors(self) -> int:
        """Tests that raised a non-assertion exception."""
        return sum(1 for r in self.per_test if r not in (_PASS, _ASSERT))

    @property
    def all_passed(self) -> bool:
        return self.loaded and self.n_tests > 0 and self.n_passed == self.n_tests

    @property
    def no_error(self) -> bool:
        """True if the code loaded and no test hit a non-assertion runtime error."""
        if not self.loaded:
            return False
        return self.n_runtime_errors == 0


def _harness_script(setup: str, code: str, tests: List[str]) -> str:
    return (
        "import ast, json, sys\n"
        f"SETUP = {setup!r}\n"
        f"CODE = {code!r}\n"
        f"TESTS = {list(tests)!r}\n"
        "res = {'syntax_error': False, 'load_error': None, 'per_test': []}\n"
        "try:\n"
        "    ast.parse(CODE)\n"
        "except SyntaxError:\n"
        f"    res['syntax_error'] = True; print({_MARKER!r} + json.dumps(res)); sys.exit(0)\n"
        "g = {'__name__': '__mbpp__'}\n"
        "try:\n"
        "    if SETUP.strip():\n"
        "        exec(SETUP, g)\n"
        "    exec(CODE, g)\n"
        "except SyntaxError:\n"
        "    res['syntax_error'] = True\n"
        "except BaseException as e:\n"
        "    res['load_error'] = type(e).__name__\n"
        "if res['syntax_error'] or res['load_error'] is not None:\n"
        f"    print({_MARKER!r} + json.dumps(res)); sys.exit(0)\n"
        "for t in TESTS:\n"
        "    try:\n"
        "        exec(t, g)\n"
        f"        res['per_test'].append({_PASS!r})\n"
        "    except AssertionError:\n"
        f"        res['per_test'].append({_ASSERT!r})\n"
        "    except BaseException as e:\n"
        "        res['per_test'].append(type(e).__name__)\n"
        f"print({_MARKER!r} + json.dumps(res))\n"
    )


def _sandbox_env(td: str, extra_env: Optional[dict] = None) -> dict:
    """Env that keeps caches/configs inside ``td`` and avoids writing .pyc files."""
    base = dict(os.environ)
    base.update({
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": td,
        "XDG_CACHE_HOME": os.path.join(td, "xdg_cache"),
        "XDG_CONFIG_HOME": os.path.join(td, "xdg_config"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONWARNINGS": "ignore",
    })
    if extra_env:
        base.update(extra_env)
    return base


def run_mbpp(
    code: str,
    test_list: List[str],
    test_setup_code: str = "",
    timeout_seconds: float = 10.0,
    python_executable: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> MBPPRunResult:
    """Run ``code`` against ``test_list`` (with optional ``test_setup_code``) in a fresh
    sandboxed subprocess and classify the outcome. Never raises on solution errors."""
    n_tests = len(test_list)
    if not (code or "").strip():
        return MBPPRunResult(load_error="EmptySolution", n_tests=n_tests)

    script = _harness_script(test_setup_code or "", code, list(test_list))
    python_executable = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="mbpp_") as td:
        path = os.path.join(td, "harness.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        env = _sandbox_env(td, extra_env)
        try:
            proc = subprocess.run(
                [python_executable, "-B", path],
                check=False, capture_output=True, text=True,
                timeout=timeout_seconds, env=env, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return MBPPRunResult(timeout=True, n_tests=n_tests)

    line = None
    for ln in (proc.stdout or "").splitlines():
        if ln.startswith(_MARKER):
            line = ln[len(_MARKER):]
    if line is None:
        # No verdict marker: a hard crash (e.g. segfault, os._exit) in the child.
        return MBPPRunResult(crashed=True, n_tests=n_tests)
    data = json.loads(line)
    return MBPPRunResult(
        syntax_error=bool(data.get("syntax_error")),
        load_error=data.get("load_error"),
        per_test=list(data.get("per_test") or []),
        n_tests=n_tests,
    )
