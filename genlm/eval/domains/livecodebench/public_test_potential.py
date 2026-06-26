"""Public-test potential and self-repair feedback. Runs the public tests on a
finished generation and turns the result into a soft score plus feedback for a
repair turn. prefix() does nothing, since a partial answer might still come good
and we never want to kill it; complete() lowers the weight when public tests
fail but never returns -inf. It only ever runs the public tests, never the
private ones."""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from genlm.control import Potential

from genlm.eval.domains.livecodebench.harness import check_correctness
from genlm.eval.domains.livecodebench.prompts import (
    decode_context,
    extract_code,
    format_lcb_prompt,
)
from genlm.eval.domains.livecodebench.runtime_execution import fork_semaphore


@dataclass
class PublicTestResult:
    """Outcome of running a single public test."""

    index: int
    input: str
    expected: str
    passed: bool
    error_message: str = ""
    got: str = ""


@dataclass
class PublicTestFeedback:
    """Aggregate public-test outcome for a generation, with a repair summary."""

    n_public: int
    n_passed: int
    results: List[PublicTestResult] = field(default_factory=list)
    # Global timeout / crash (overload, not the code): not cached, so retries re-run.
    transient: bool = False

    @property
    def all_passed(self) -> bool:
        return self.n_public > 0 and self.n_passed == self.n_public

    @property
    def n_failed(self) -> int:
        return self.n_public - self.n_passed

    @property
    def pass_fraction(self) -> float:
        return self.n_passed / self.n_public if self.n_public else 1.0

    def summary(self, max_cases: int = 3, max_chars: int = 300) -> str:
        """Human-readable failing-test report for a repair prompt."""
        if self.n_public == 0:
            return "No public tests were available."
        if self.all_passed:
            return f"All {self.n_public} public tests passed."

        def trim(s: str) -> str:
            s = str(s)
            return s if len(s) <= max_chars else s[:max_chars] + "...[truncated]"

        lines = [f"Passed {self.n_passed}/{self.n_public} public tests. Failing cases:"]
        shown = [r for r in self.results if not r.passed][:max_cases]
        for r in shown:
            lines.append(f"- Input:\n{trim(r.input)}")
            lines.append(f"  Expected output:\n{trim(r.expected)}")
            if r.got:
                lines.append(f"  Your output:\n{trim(r.got)}")
            if r.error_message:
                lines.append(f"  Error: {trim(r.error_message)}")
        return "\n".join(lines)


class LCBPublicTestPotential(Potential):
    """Soft public-test verifier. ``prefix`` never kills; ``complete`` returns
    0.0 when all public tests pass, otherwise a finite penalty proportional to
    the number of failed tests (floored at ``min_score``, never -inf)."""

    def __init__(
        self,
        vocabulary=None,
        public_eval_sample: Optional[dict] = None,
        timeout_seconds: float = 6.0,
        penalty_per_failed: float = 2.0,
        min_score: float = -10.0,
        extraction_style: str = "generic",
        max_total_seconds: Optional[float] = None,
        f: Optional[Callable[[List[bytes]], List[bytes]]] = None,
    ):
        vocabulary = vocabulary or [bytes([i]) for i in range(256)]
        super().__init__(vocabulary=vocabulary)
        self.public_eval_sample = public_eval_sample or {}
        self.timeout_seconds = float(timeout_seconds)
        self.penalty_per_failed = abs(float(penalty_per_failed))
        self.min_score = float(min_score)
        self.extraction_style = extraction_style
        self.max_total_seconds = max_total_seconds
        self.f = f

        io = {}
        if self.public_eval_sample.get("input_output"):
            try:
                io = json.loads(self.public_eval_sample["input_output"])
            except (TypeError, ValueError):
                io = {}
        self._inputs = list(io.get("inputs") or [])
        self._outputs = list(io.get("outputs") or [])
        self._fn_name = io.get("fn_name")
        self._cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 2048

    def coerce(self, other, f=None, prune=True):
        return LCBPublicTestPotential(
            vocabulary=list(other.vocab),
            public_eval_sample=self.public_eval_sample,
            timeout_seconds=self.timeout_seconds,
            penalty_per_failed=self.penalty_per_failed,
            min_score=self.min_score,
            extraction_style=self.extraction_style,
            max_total_seconds=self.max_total_seconds,
            f=f,
        )

    async def prefix(self, context: List[bytes]) -> float:
        return 0.0  # runs only at the end of generation; never kills a prefix

    async def complete(self, context: List[bytes]) -> float:
        if self.f is not None:
            context = self.f(context)
        code = self._extract(decode_context(context))
        if not self._inputs:
            return 0.0
        feedback = self._cache.get(code)
        if feedback is None:
            # Fork off the event loop, bounded; cache writes stay in the
            # coroutine so concurrent particles don't race on the dict.
            async with fork_semaphore():
                feedback = await asyncio.to_thread(self._evaluate, code)
            if not feedback.transient:
                self._store(code, feedback)
        else:
            self._cache.move_to_end(code)
        return self._score(feedback)

    def _score(self, feedback: "PublicTestFeedback") -> float:
        if feedback.n_public == 0 or feedback.all_passed:
            return 0.0
        return max(self.min_score, -self.penalty_per_failed * feedback.n_failed)

    def run_public_tests(self, generation: str) -> PublicTestFeedback:
        """Run every public test on ``generation`` (a full model output; code is
        extracted with the configured style) and return structured feedback.
        Results are cached per extracted code."""
        if not self._inputs:
            return PublicTestFeedback(n_public=0, n_passed=0)
        code = self._extract(generation)
        feedback = self._cache.get(code)
        if feedback is None:
            feedback = self._evaluate(code)
            if not feedback.transient:
                self._store(code, feedback)
        else:
            self._cache.move_to_end(code)
        return feedback

    def _extract(self, generation: str) -> str:
        # Extract exactly as the evaluator (no fallback for unfenced output).
        return extract_code(generation, self.extraction_style)

    def _store(self, code: str, feedback: "PublicTestFeedback") -> None:
        self._cache[code] = feedback
        self._cache.move_to_end(code)
        if len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

    def _evaluate(self, code: str) -> PublicTestFeedback:
        # One test per run so a failure does not hide later ones (the grader
        # short-circuits within a sample on the first miss).
        results: List[PublicTestResult] = []
        n_passed, transient = 0, False
        for i, (inp, out) in enumerate(zip(self._inputs, self._outputs)):
            sample = {"input_output": json.dumps(
                {"inputs": [inp], "outputs": [out], "fn_name": self._fn_name})}
            res, meta = check_correctness(
                sample, code, timeout=self.timeout_seconds,
                max_total_seconds=self.max_total_seconds)
            transient = transient or (-1 in res)  # global timeout / crash, not the code
            passed = bool(res) and all(r > 0 for r in res)
            n_passed += int(passed)
            results.append(PublicTestResult(
                index=i, input=inp, expected=out, passed=passed,
                error_message="" if passed else str(
                    meta.get("error_message") or meta.get("error") or ""),
                got="" if passed else str(meta.get("output", "")),
            ))
        return PublicTestFeedback(
            n_public=len(results), n_passed=n_passed, results=results,
            transient=transient)


def repair_question_content(question_content: str, previous_code: str,
                            feedback: PublicTestFeedback) -> str:
    """Augment the original question with the failed attempt and public-test
    feedback, for a second (repair) generation turn."""
    return (
        f"{question_content}\n\n"
        f"A previous attempt produced this program:\n"
        f"```python\n{previous_code}\n```\n\n"
        f"{feedback.summary()}\n\n"
        f"Fix the program so it passes all tests."
    )


def format_repair_prompt(tokenizer, instance, previous_generation: str,
                         feedback: PublicTestFeedback, use_chat_format: bool = False,
                         style: str = "generic"):
    """Build the next-turn repair prompt for ``instance`` and return token ids,
    matching the contract of ``default_prompt_formatter``."""
    previous_code = extract_code(previous_generation, style)
    question = repair_question_content(instance.question_content, previous_code, feedback)
    row = {"question_content": question, "starter_code": instance.starter_code}
    text = format_lcb_prompt(row, tokenizer=tokenizer, chat_template=use_chat_format, style=style)
    from genlm.eval.domains.livecodebench.prompts import RAW_STYLES

    if style in RAW_STYLES:
        return tokenizer.encode(text)
    return tokenizer.encode(text, add_special_tokens=not use_chat_format)
