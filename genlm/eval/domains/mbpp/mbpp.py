"""MBPP domain: dataset loader, evaluator, and prompt formatter.

MBPP (Mostly Basic Python Problems, Austin et al. 2021) -- each problem is a short natural
language task plus a handful of ``assert`` tests that call the target function by name. We
load ``google-research-datasets/mbpp`` (configs ``full`` and ``sanitized``), build a prompt
from the description + tests, and grade by running the tests (see :mod:`.execution`).
"""
from __future__ import annotations

import logging
import random
import sys
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type

from genlm.eval.core import Dataset, EvaluationResult, Evaluator, Instance
from genlm.eval.domains.mbpp.execution import extract_code, run_mbpp

log = logging.getLogger(__name__)


#############
# MBPP Data #
#############

class MBPPInstance(Instance):
    """Schema for one MBPP problem."""

    text: str                                  # natural-language task description
    test_list: List[str]                       # assert statements calling the target function
    prompt: str                                # the LM prompt (description + tests)
    test_setup_code: str = ""                  # imports / setup exec'd before the solution
    challenge_test_list: List[str] = []        # extra harder tests (not used for grading by default)
    reference_code: Optional[str] = None       # the dataset's gold solution
    config: str = "full"                       # "full" or "sanitized"


def build_prompt(text: str, test_list: Sequence[str]) -> str:
    """Standard MBPP prompt: task description followed by the tests the code must pass.

    Including the asserts is essential -- they pin the exact function name and signature the
    solution must define.
    """
    tests = "\n".join(test_list)
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text.strip()}\n"
        "Your code should pass these tests:\n\n"
        f"{tests}\n"
    )


class MBPPDataset(Dataset[MBPPInstance]):
    """Dataset for MBPP evaluation."""

    def __init__(self, rows: List[Mapping[str, Any]], config: str = "full"):
        self._rows = list(rows)
        self._config = config

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[MBPPInstance]:
        for row in self._rows:
            text = str(row.get("text") or row.get("prompt") or "")
            test_list = list(row.get("test_list") or [])
            # ``full`` has a single test_setup_code string; ``sanitized`` has test_imports (list).
            setup = row.get("test_setup_code")
            if setup is None:
                setup = "\n".join(row.get("test_imports") or [])
            yield MBPPInstance(
                instance_id=int(row["task_id"]),
                text=text,
                test_list=test_list,
                test_setup_code=str(setup or ""),
                challenge_test_list=list(row.get("challenge_test_list") or []),
                reference_code=row.get("code"),
                prompt=build_prompt(text, test_list),
                config=self._config,
            )

    @property
    def schema(self) -> Type[MBPPInstance]:
        return MBPPInstance

    @classmethod
    def from_hf(
        cls,
        split: str = "test",
        config: str = "full",
        max_instances: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        cache_dir: Optional[str] = None,
    ) -> "MBPPDataset":
        """Load MBPP from Hugging Face (``google-research-datasets/mbpp``).

        ``config`` is ``"full"`` (974 problems) or ``"sanitized"`` (427, hand-verified).
        The canonical evaluation split is ``"test"``.
        """
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/mbpp", config, split=split, cache_dir=cache_dir)
        rows: List[Mapping[str, Any]] = list(ds)
        if shuffle:
            random.Random(seed).shuffle(rows)
        if isinstance(max_instances, int) and max_instances >= 0:
            rows = rows[:max_instances]
        log.info("Loaded MBPP: %d instances (config=%s split=%s)", len(rows), config, split)
        return cls(rows, config=config)


#############
# Evaluator #
#############

class MBPPEvaluator(Evaluator[MBPPInstance]):
    """Grade a generation by running the MBPP tests: score 1.0 iff every test passes."""

    def __init__(
        self,
        python_executable: Optional[str] = None,
        timeout_seconds: float = 10.0,
        extra_env: Optional[Dict[str, str]] = None,
        include_challenge_tests: bool = False,
        max_log_chars: int = 4000,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.timeout_seconds = float(timeout_seconds)
        self.extra_env = dict(extra_env or {})
        self.include_challenge_tests = bool(include_challenge_tests)
        self.max_log_chars = int(max_log_chars)

    def evaluate_sample(self, instance: MBPPInstance, response: str) -> EvaluationResult:
        code = extract_code(response)
        tests = list(instance.test_list)
        if self.include_challenge_tests:
            tests += list(instance.challenge_test_list)
        meta: Dict[str, Any] = {"task_id": instance.instance_id, "config": instance.config}
        if not code.strip():
            return EvaluationResult(score=0.0, desc="empty solution", metadata=meta)

        result = run_mbpp(
            code, tests, instance.test_setup_code,
            timeout_seconds=self.timeout_seconds,
            python_executable=self.python_executable,
            extra_env=self.extra_env,
        )
        meta.update({
            "n_tests": result.n_tests, "n_passed": result.n_passed,
            "syntax_error": result.syntax_error, "load_error": result.load_error,
            "timeout": result.timeout,
        })
        desc = code if len(code) <= self.max_log_chars else code[: self.max_log_chars] + "\n...[truncated]"
        return EvaluationResult(score=1.0 if result.all_passed else 0.0, desc=desc, metadata=meta)


##########################
# Prompt formatter (LM)  #
##########################

def default_prompt_formatter(
    tokenizer,
    instance: MBPPInstance,
    use_chat_format: bool = False,  # conform with the evaluator interface
) -> List[int]:
    return tokenizer.encode(instance.prompt)
