"""Evaluator for multilingual LiveCodeBench.

Extracts the code block from a generation and grades it against the problem's stdin/stdout
tests via a ``MultilingualCodeExecutor``. Strict 0/1, memoized on (instance_id, code).
``grading="lenient"`` is Multi-LCB's comparator; ``"exact"`` is Agnostics rstrip-equality.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from genlm.eval.core import EvaluationResult, Evaluator

from .dataset import MultilingualLCBInstance
from .executor import LocalSubprocessExecutor, MultilingualCodeExecutor
from .prompts import extract_code  # Multi-LCB-matching extractor (first block)


class MultilingualLCBEvaluator(Evaluator[MultilingualLCBInstance]):
    def __init__(
        self,
        timeout_seconds: float = 6.0,
        executor: Optional[MultilingualCodeExecutor] = None,
        max_log_chars: int = 4000,
        grading: str = "lenient",
    ):
        self.timeout_seconds = float(timeout_seconds)
        # grading picks the default executor's comparator; ignored if a custom executor is passed.
        self.executor = executor or LocalSubprocessExecutor(grading=grading)
        self.max_log_chars = int(max_log_chars)
        self._cache: Dict[Tuple[Any, str], Tuple[bool, Dict[str, Any]]] = {}

    def _meta(self, instance: MultilingualLCBInstance, **extra: Any) -> Dict[str, Any]:
        return {
            "question_id": instance.question_id,
            "language": instance.language,
            "difficulty": instance.difficulty,
            **extra,
        }

    def evaluate_sample(
        self, instance: MultilingualLCBInstance, response: str
    ) -> EvaluationResult:
        code = extract_code(response)
        if not code:
            return EvaluationResult(
                score=0.0, desc="empty code", metadata=self._meta(instance)
            )
        if not instance.eval_sample or "input_output" not in instance.eval_sample:
            return EvaluationResult(
                score=0.0, desc="missing eval_sample", metadata=self._meta(instance)
            )
        try:
            io = json.loads(instance.eval_sample["input_output"])
            inputs, outputs = io["inputs"], io["outputs"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Malformed/prompts-only eval_sample scores a failure rather than crashing the run.
            return EvaluationResult(
                score=0.0, desc="malformed eval_sample", metadata=self._meta(instance)
            )
        if not inputs:
            # No tests: score a failure instead of crashing in the vendored compile_and_run.
            return EvaluationResult(
                score=0.0, desc="no test inputs", metadata=self._meta(instance)
            )

        lang = instance.language
        self.executor.prepare(lang)  # idempotent; raises if the toolchain is missing

        key = (instance.instance_id, code)
        if key not in self._cache:
            self._cache[key] = self.executor.run(
                code, inputs, outputs, lang, self.timeout_seconds
            )
        solved, run_meta = self._cache[key]

        desc = (
            code
            if len(code) <= self.max_log_chars
            else code[: self.max_log_chars] + "\n...[truncated]"
        )
        return EvaluationResult(
            score=1.0 if solved else 0.0,
            desc=desc,
            metadata=self._meta(instance, status=run_meta.get("status")),
        )
