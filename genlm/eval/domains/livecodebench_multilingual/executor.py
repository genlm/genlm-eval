"""Execution backends for multilingual LiveCodeBench.

A backend grades one (code, language) candidate against shared stdin/stdout tests. The only
backend so far is ``LocalSubprocessExecutor``, which wraps the vendored Multi-LCB executor
(``testing_plang.eval_plang_code``). A future container backend can implement the same
protocol.
"""

import shutil
import subprocess
from math import ceil
from typing import Any, Dict, List, Protocol, Tuple

from .vendored import testing_plang as _tp

# Binaries each language needs on PATH at compile/run time. Probed so a missing toolchain
# fails fast with a clear message instead of a cryptic FileNotFoundError from Popen. "python"
# is the running interpreter (always present), so it has no probe.
_TOOLCHAIN: Dict[str, List[str]] = {
    "python": [],
    "c++": ["g++"],
    "java": ["javac", "java"],
    "c#": ["mcs", "mono"],
    "go": ["go"],
    "javascript": ["node"],
    # TS is graded Deno-native (fixtures and prompts target Deno, which runs .ts directly).
    # node-style TS still grades if tsc/node/npm happen to be present, but deno is what we require.
    "typescript": ["deno"],
    "rust": ["rustc"],
    "ruby": ["ruby"],
    "php": ["php"],
    "kotlin": ["kotlinc", "java"],
    "scala": ["scalac", "scala"],
    "lua": ["luajit"],
    "julia": ["julia"],
    "r": ["Rscript"],
    "ocaml": ["ocaml"],
    "fortran": ["gfortran"],
}


def is_toolchain_available(language: str) -> bool:
    """True if every binary ``language`` needs is on PATH (python is always available)."""
    probes = _TOOLCHAIN.get(language)
    if probes is None:
        return False
    return all(shutil.which(b) is not None for b in probes)


class MultilingualCodeExecutor(Protocol):
    def prepare(self, language: str) -> None:
        """One-time per-language setup before grading a batch (idempotent)."""
        ...  # pragma: no cover

    def run(
        self,
        code: str,
        inputs: List[str],
        outputs: List[str],
        language: str,
        timeout: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Return (solved, metadata) for one candidate against the stdin/stdout tests."""
        ...  # pragma: no cover


class LocalSubprocessExecutor:
    """Grade candidates by compiling/running them locally via the vendored executor.

    No container isolation: generated code runs as host subprocesses with only rlimit +
    process-group SIGKILL. Run on a dedicated/disposable node only.
    """

    def __init__(self, grading: str = "lenient") -> None:
        if grading not in ("lenient", "exact"):
            raise ValueError("grading must be 'lenient' or 'exact'")
        # "exact" = Agnostics-style whole-output rstrip equality; "lenient" = Multi-LCB's
        # per-line + float-tolerant comparator (default, leaderboard-comparable).
        self.exact_match = grading == "exact"
        self._prepared: set[str] = set()

    def prepare(self, language: str) -> None:
        if language in self._prepared:
            return
        if language not in _tp.eval_scripts:
            raise NotImplementedError(
                f"language {language!r} is not yet wired in the vendored executor "
                f"(eval_scripts has {sorted(_tp.eval_scripts)})"
            )
        if not is_toolchain_available(language):
            raise RuntimeError(
                f"toolchain for {language!r} not found on PATH "
                f"(need {_TOOLCHAIN.get(language)}); install it or skip this language"
            )
        # We deliberately do not run `go clean -cache` here: clearing it forces a cold stdlib
        # rebuild that can exceed the 60s build timeout, while a warm cache builds fast. A very
        # long go run that fills the cache may need a periodic manual `go clean -cache`.
        if language == "julia":
            # Warm Julia's precompile cache once: the first cold run builds it and can exceed
            # a per-test timeout, while warm runs start in well under a second.
            try:
                subprocess.run(
                    ["julia", "--startup-file=no", "-e", "1+1"],
                    capture_output=True,
                    timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._prepared.add(language)

    def run(
        self,
        code: str,
        inputs: List[str],
        outputs: List[str],
        language: str,
        timeout: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        if language not in _tp.eval_scripts:
            raise NotImplementedError(
                f"language {language!r} is not yet wired in the vendored executor "
                f"(eval_scripts has {sorted(_tp.eval_scripts)})"
            )
        # max(1, ...): eval_plang_code passes this to subprocess.communicate(timeout=...); a 0
        # would make every test time out instantly (mirrors livecodebench/harness.py).
        scores, meta = _tp.eval_plang_code(
            code,
            list(inputs),
            list(outputs),
            language,
            max(1, int(ceil(timeout))),
            exact_match=self.exact_match,
        )
        # Solved iff every per-test score is positive (PASSED=1). On any failure
        # eval_plang_code returns a short list with a non-positive score, so all(>0) already
        # short-circuits. EXECFAIL returns [TestScore.EXECFAIL].
        solved = bool(scores) and all(s.value > 0 for s in scores)
        metadata = {
            "status": str(getattr(meta, "error", "ok")),
            "per_test": [s.name for s in scores],
            "n_tests": len(outputs),
        }
        return solved, metadata
