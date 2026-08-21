"""MBPP test-passing potential: a soft verifier over the problem's ``assert`` tests.

``prefix()`` never kills (a partial answer might still come good); ``complete()`` returns
0.0 when every test passes, otherwise a finite penalty proportional to the number of failed
tests (floored at ``min_score``, never -inf), mirroring the LiveCodeBench public-test
potential. Set ``hard=True`` for an all-or-nothing 0.0 / -inf verifier instead.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, List, Optional

from genlm.control import Potential

from genlm.eval.domains.mbpp.execution import (
    MBPPRunResult,
    extract_code,
    run_mbpp,
    subprocess_semaphore,
)


@dataclass
class TestPassingFeedback:
    """Aggregate test outcome for a generation."""

    n_tests: int
    n_passed: int
    syntax_error: bool = False
    load_error: Optional[str] = None
    # Global timeout / hard crash (overload, not the code): not cached, so retries re-run.
    transient: bool = False

    @property
    def all_passed(self) -> bool:
        return self.n_tests > 0 and self.n_passed == self.n_tests and not self.transient

    @property
    def n_failed(self) -> int:
        return self.n_tests - self.n_passed


class MBPPTestPassingPotential(Potential):
    """Soft test verifier. ``prefix`` never kills; ``complete`` returns 0.0 when all tests
    pass, else a finite penalty proportional to the number of failed tests (never -inf,
    unless ``hard=True``)."""

    def __init__(
        self,
        vocabulary=None,
        test_list: Optional[List[str]] = None,
        test_setup_code: str = "",
        timeout_seconds: float = 10.0,
        python_executable: Optional[str] = None,
        penalty_per_failed: float = 2.0,
        min_score: float = -10.0,
        hard: bool = False,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.test_list = list(test_list or [])
        self.test_setup_code = test_setup_code or ""
        self.timeout_seconds = float(timeout_seconds)
        self.python_executable = python_executable
        self.penalty_per_failed = abs(float(penalty_per_failed))
        self.min_score = float(min_score)
        self.hard = bool(hard)
        self.f = f
        self._cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 2048

    def coerce(self, other, f=None, prune=True):
        return MBPPTestPassingPotential(
            vocabulary=list(other.vocab),
            test_list=self.test_list,
            test_setup_code=self.test_setup_code,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            penalty_per_failed=self.penalty_per_failed,
            min_score=self.min_score,
            hard=self.hard,
            f=f,
        )

    async def prefix(self, context: List[bytes]) -> float:
        return 0.0  # runs only at the end of generation; never kills a prefix

    async def complete(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = self._decode(context)
        if not self.test_list:
            return 0.0
        feedback = self._cache.get(code)
        if feedback is None:
            async with subprocess_semaphore():
                feedback = await asyncio.to_thread(self._evaluate, code)
            if not feedback.transient:
                self._store(code, feedback)
        else:
            self._cache.move_to_end(code)
        return self._score(feedback)

    def run_tests(self, generation: str) -> TestPassingFeedback:
        """Run every test on a full model output (code extracted with the default style)."""
        code = extract_code(generation)
        feedback = self._cache.get(code)
        if feedback is None:
            feedback = self._evaluate(code)
            if not feedback.transient:
                self._store(code, feedback)
        else:
            self._cache.move_to_end(code)
        return feedback

    def _score(self, feedback: TestPassingFeedback) -> float:
        if feedback.n_tests == 0 or feedback.all_passed:
            return 0.0
        if self.hard:
            return float("-inf")
        return max(self.min_score, -self.penalty_per_failed * feedback.n_failed)

    def _decode(self, context) -> str:
        if isinstance(context, str):
            raw = context
        elif isinstance(context, bytes):
            raw = context.decode("utf-8", errors="ignore")
        else:
            pieces = []
            for tok in context or []:
                if isinstance(tok, int):
                    pieces.append(bytes([tok]))
                elif isinstance(tok, bytes):
                    pieces.append(tok)
                else:
                    pieces.append(str(tok).encode("utf-8", errors="ignore"))
            raw = b"".join(pieces).decode("utf-8", errors="ignore")
        return extract_code(raw)

    def _evaluate(self, code: str) -> TestPassingFeedback:
        if not code.strip():
            return TestPassingFeedback(n_tests=len(self.test_list), n_passed=0,
                                       load_error="EmptySolution")
        result: MBPPRunResult = run_mbpp(
            code, self.test_list, self.test_setup_code,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
        )
        return TestPassingFeedback(
            n_tests=result.n_tests, n_passed=result.n_passed,
            syntax_error=result.syntax_error, load_error=result.load_error,
            transient=result.timeout or result.crashed,
        )

    def _store(self, code: str, feedback: TestPassingFeedback) -> None:
        self._cache[code] = feedback
        self._cache.move_to_end(code)
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)
