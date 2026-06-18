"""Runtime no-error potential: scores 0.0 if the extracted code runs without
raising and -inf if it does not. Wrong answers are fine, only crashes matter,
like the DS-1000 potential. prefix() runs the leading statements of the partial
code against a public input; complete() runs the whole solution against all the
public inputs. With no public inputs it only checks syntax. Runs in a forked
child."""
from __future__ import annotations

import ast
import asyncio
import codeop
import json
from collections import OrderedDict
from typing import Callable, List, Optional

from genlm.control import Potential

from genlm.eval.domains.livecodebench.prompts import (
    decode_context,
    extract_code,
    extract_code_prefix,
)
from genlm.eval.domains.livecodebench.runtime_execution import (
    OK,
    SYNTAX,
    TIMEOUT,
    fork_semaphore,
    run_noerror_check,
)


def _syntax_status(code: str) -> str:
    """complete (parses), incomplete (a continuation can still fix it), or broken."""
    if code.endswith("\\\n"):  # codeop misreads a trailing line-continuation as broken
        return "incomplete"
    try:
        ast.parse(code)
        return "complete"
    except SyntaxError:
        pass
    try:
        codeop.compile_command(code, "<prefix>", "exec")
        return "incomplete"  # compiled or returned None: a continuation may still close it
    except (SyntaxError, ValueError, OverflowError):
        return "broken"


class LCBRuntimeNoErrorPotential(Potential):
    """0.0 if the extracted code runs without error, -inf otherwise; wrong
    answers are tolerated. Uses only the public test inputs, never the held-out
    private ones."""

    def __init__(
        self,
        vocabulary=None,
        public_eval_sample: Optional[dict] = None,
        timeout_seconds: float = 6.0,
        max_inputs: int = 1,
        extraction_style: str = "generic",
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.public_eval_sample = public_eval_sample or {}
        self.timeout_seconds = float(timeout_seconds)
        self.max_inputs = int(max_inputs)
        self.extraction_style = extraction_style
        self.f = f
        self.last_was_syntax_error = False

        io = {}
        if self.public_eval_sample.get("input_output"):
            try:
                io = json.loads(self.public_eval_sample["input_output"])
            except (TypeError, ValueError):
                io = {}
        # Public inputs only, no outputs, so nothing leaks. prefix() uses a
        # subset of complete()'s inputs, which keeps the soundness invariant.
        self._inputs = list(io.get("inputs") or [])
        self._prefix_inputs = self._inputs[: self.max_inputs]
        self._fn_name = io.get("fn_name")

        # Prefixes extending a hung one re-run the same statements: defer them.
        self._hung_prefixes: List[str] = []
        self._hung_prefixes_maxsize = 16
        # SMC clones particles into repeated prefixes; cache verdicts per code.
        self._score_cache: OrderedDict = OrderedDict()
        self._score_cache_maxsize = 4096
        self.cache_hits = 0
        self.cache_misses = 0

    def coerce(self, other, f=None, prune=True):
        return LCBRuntimeNoErrorPotential(
            vocabulary=list(other.vocab),
            public_eval_sample=self.public_eval_sample,
            timeout_seconds=self.timeout_seconds,
            max_inputs=self.max_inputs,
            extraction_style=self.extraction_style,
            f=f,
        )

    async def prefix(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        raw = decode_context(context)
        # Newline guardrail: only judge at line boundaries (default line sampler).
        if not raw.endswith("\n"):
            return 0.0
        code = extract_code_prefix(raw, self.extraction_style)
        if not code.strip():
            self.last_was_syntax_error = False
            return 0.0
        status = _syntax_status(code)
        if status == "incomplete":
            self.last_was_syntax_error = False
            return 0.0
        if status == "broken":
            self.last_was_syntax_error = True
            return float("-inf")
        if self._fn_name:
            # Functional: only complete() calls the entrypoint. Calling it on a
            # prefix is unsound (a helper it needs may be defined later, then dropped).
            self.last_was_syntax_error = False
            return 0.0
        if not self._prefix_inputs:  # syntax-only mode: parseable prefix, nothing to run
            self.last_was_syntax_error = False
            return 0.0
        for hung in self._hung_prefixes:
            if code.startswith(hung):
                return 0.0
        return await self._run(code, mode="prefix")

    async def complete(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = extract_code(decode_context(context), self.extraction_style)
        if not code.strip():
            self.last_was_syntax_error = False
            return float("-inf")  # an empty completion is not a runnable solution
        status = _syntax_status(code)
        if status in ("broken", "incomplete"):  # at EOS, unparseable code is fatal
            self.last_was_syntax_error = True
            return float("-inf")
        if not self._inputs:
            self.last_was_syntax_error = False
            return 0.0
        return await self._run(code, mode="complete")

    async def _run(self, code: str, mode: str) -> float:
        key = (mode, code)
        cached = self._score_cache.get(key)
        if cached is not None:
            self._score_cache.move_to_end(key)
            self.cache_hits += 1
            value, syntax_error = cached
            self.last_was_syntax_error = syntax_error
            return value
        self.cache_misses += 1

        # prefix: a capped input subset (cheap); complete: all (a later input may crash).
        inputs = self._prefix_inputs if mode == "prefix" else self._inputs
        async with fork_semaphore():
            verdict = await asyncio.to_thread(
                run_noerror_check, code, inputs, self._fn_name,
                mode == "prefix", self.timeout_seconds,
            )

        if verdict == TIMEOUT:
            # A slow prefix may still finish: defer (uncached). complete kills.
            self.last_was_syntax_error = False
            if mode == "prefix":
                self._hung_prefixes.append(code)
                if len(self._hung_prefixes) > self._hung_prefixes_maxsize:
                    self._hung_prefixes.pop(0)
                return 0.0
            return float("-inf")

        self.last_was_syntax_error = verdict == SYNTAX
        value = 0.0 if verdict == OK else float("-inf")
        self._score_cache[key] = (value, self.last_was_syntax_error)
        self._score_cache.move_to_end(key)
        if len(self._score_cache) > self._score_cache_maxsize:
            self._score_cache.popitem(last=False)
        return value
