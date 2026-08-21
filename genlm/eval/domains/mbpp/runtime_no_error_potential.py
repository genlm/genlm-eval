"""MBPP runtime-no-error potential: 0.0 if the code runs on the tests without a runtime
error, -inf otherwise. A failed ``assert`` (wrong answer) still counts as no-error -- only
crashes, exceptions, timeouts and syntax errors are penalised, mirroring the DS-1000 and
LiveCodeBench no-error potentials.

``prefix()`` only kills a partial generation whose code is already syntactically unfixable
(running the asserts mid-stream is unsound -- the function may not be defined yet), so it
otherwise defers with 0.0. ``complete()`` runs the whole solution against every test.
"""
from __future__ import annotations

import ast
import asyncio
import codeop
from collections import OrderedDict
from typing import Callable, List, Optional

from genlm.control import Potential

from genlm.eval.domains.mbpp.execution import (
    extract_code,
    extract_code_prefix,
    run_mbpp,
    subprocess_semaphore,
)


def _syntax_status(code: str) -> str:
    """complete (parses), incomplete (a continuation can still fix it), or broken."""
    if code.endswith("\\\n"):  # trailing line-continuation: a later line completes it
        return "incomplete"
    try:
        ast.parse(code)
        return "complete"
    except SyntaxError:
        pass
    try:
        codeop.compile_command(code, "<prefix>", "exec")
        return "incomplete"
    except (SyntaxError, ValueError, OverflowError):
        return "broken"


def _decode(context) -> str:
    """Normalise a genlm.control context (bytes tokens / ints / str) to a string."""
    if isinstance(context, str):
        return context
    if isinstance(context, bytes):
        return context.decode("utf-8", errors="ignore")
    pieces = []
    for tok in context or []:
        if isinstance(tok, int):
            pieces.append(bytes([tok]))
        elif isinstance(tok, bytes):
            pieces.append(tok)
        else:
            pieces.append(str(tok).encode("utf-8", errors="ignore"))
    return b"".join(pieces).decode("utf-8", errors="ignore")


class MBPPRuntimeNoErrorPotential(Potential):
    """0.0 if the extracted code runs against the MBPP tests without a runtime error,
    -inf otherwise. Wrong answers (failed asserts) are tolerated."""

    def __init__(
        self,
        vocabulary=None,
        test_list: Optional[List[str]] = None,
        test_setup_code: str = "",
        timeout_seconds: float = 10.0,
        python_executable: Optional[str] = None,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.test_list = list(test_list or [])
        self.test_setup_code = test_setup_code or ""
        self.timeout_seconds = float(timeout_seconds)
        self.python_executable = python_executable
        self.f = f
        self.last_was_syntax_error = False
        # SMC clones particles into repeated prefixes; cache verdicts per exact code.
        self._score_cache: OrderedDict = OrderedDict()
        self._score_cache_maxsize = 4096

    def coerce(self, other, f=None, prune=True):
        return MBPPRuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            test_list=self.test_list,
            test_setup_code=self.test_setup_code,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            f=f,
        )

    async def prefix(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        raw = _decode(context)
        # Newline guardrail: only judge at line boundaries (default line sampler).
        if not raw.endswith("\n"):
            return 0.0
        code = extract_code_prefix(raw)
        if not code.strip():
            self.last_was_syntax_error = False
            return 0.0
        status = _syntax_status(code)
        if status == "broken":
            self.last_was_syntax_error = True
            return float("-inf")
        # complete/incomplete: parseable so far, or fixable by a continuation. Running the
        # asserts now is unsound (the target function may be defined later), so defer.
        self.last_was_syntax_error = False
        return 0.0

    async def complete(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = extract_code(_decode(context))
        if not code.strip():
            self.last_was_syntax_error = False
            return float("-inf")  # an empty completion is not a runnable solution
        if _syntax_status(code) == "broken":
            self.last_was_syntax_error = True
            return float("-inf")
        if not self.test_list:
            self.last_was_syntax_error = False
            return 0.0  # nothing to run it against: syntax-only check
        return await self._score(code)

    async def _score(self, code: str) -> float:
        cached = self._score_cache.get(code)
        if cached is not None:
            self._score_cache.move_to_end(code)
            self.last_was_syntax_error = cached[1]
            return cached[0]

        async with subprocess_semaphore():
            result = await asyncio.to_thread(
                run_mbpp, code, self.test_list, self.test_setup_code,
                self.timeout_seconds, self.python_executable,
            )
        if result.timeout or result.crashed:
            # Transient (overload / hard crash), not a clean code verdict: don't cache.
            self.last_was_syntax_error = False
            return float("-inf")

        self.last_was_syntax_error = result.syntax_error
        value = 0.0 if result.no_error else float("-inf")
        self._score_cache[code] = (value, self.last_was_syntax_error)
        self._score_cache.move_to_end(code)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value
