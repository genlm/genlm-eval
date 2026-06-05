"""LiveCodeBench (Jain et al., 2024) domain for genlm-eval.

Mirrors the ds1000 domain: a pydantic ``Instance``, a ``Dataset`` with ``from_hf``
(+ ``from_jsonl``) loaders, and an ``Evaluator`` that scores a generation by running
it against the problem's test cases via the vendored official LCB harness
(``harness.passed_all``). Ported from the genlm/latent PR (jac/add-livecodebench),
which deferred "lifting the domain into genlm-eval".

Correctness signal: extract the code block from the completion, then run it against
``eval_sample['input_output']`` (stdin/stdout or call-based) in a forked child. Strict
0/1 per problem (all test cases must pass).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Type

from genlm.eval.core import Dataset, Instance
from genlm.eval.core.evaluator import EvaluationResult, Evaluator

from genlm.eval.domains.livecodebench._fetch import iter_release_rows
from genlm.eval.domains.livecodebench.harness import passed_all
from genlm.eval.domains.livecodebench.prompts import extract_code, format_lcb_prompt

################
# Data         #
################


class LiveCodeBenchInstance(Instance):
    """Schema for one LiveCodeBench problem.

    ``eval_sample`` is the harness-ready ``{"input_output": <json str>}`` payload
    (decoded test cases); ``testtype`` is ``stdin`` or ``functional``.
    """

    question_id: str
    question_content: str
    starter_code: str = ""
    difficulty: str = "unknown"
    platform: str = "unknown"
    testtype: str = "stdin"
    contest_date: str = ""
    release: str = ""
    eval_sample: Dict[str, str]
    metadata: Dict[str, Any] = {}


def _stratified_split(rows: List[dict], split: str, test_frac: float, seed: int) -> List[dict]:
    """Per-(testtype, difficulty) disjoint train/test split (ported from the latent
    loader). testtype is the biggest behavioral axis, so balancing it keeps the
    eval pass-rate honest. Returns all rows when ``split`` is None."""
    if split is None:
        return rows
    by_key: Dict[Any, list] = {}
    for r in rows:
        by_key.setdefault((r.get("testtype", "stdin"), r.get("difficulty", "unknown")), []).append(r)
    rng = random.Random(seed)
    out: List[dict] = []
    for key in sorted(by_key):
        items = list(by_key[key])
        rng.shuffle(items)
        n_test = int(round(len(items) * test_frac))
        if split == "test":
            out.extend(items[:n_test])
        elif split == "train":
            out.extend(items[n_test:])
        else:
            raise ValueError(f"split must be 'train', 'test', or None; got {split!r}")
    rng.shuffle(out)
    return out


def _filter_rows(rows: Iterable[dict], start_date: Optional[str], end_date: Optional[str],
                 difficulties: Optional[Sequence[str]], testtypes: Optional[Sequence[str]]) -> List[dict]:
    diff_set = {d.lower() for d in difficulties} if difficulties else None
    tt_set = {t.lower() for t in testtypes} if testtypes else None
    out = []
    for r in rows:
        d = (r.get("contest_date") or "")[:10]  # ISO dates sort lexically
        if start_date and (not d or d < start_date):
            continue
        if end_date and (not d or d > end_date):
            continue
        if diff_set is not None and str(r.get("difficulty", "")).lower() not in diff_set:
            continue
        if tt_set is not None and str(r.get("testtype", "")).lower() not in tt_set:
            continue
        out.append(r)
    return out


class LiveCodeBenchDataset(Dataset[LiveCodeBenchInstance]):
    """Dataset for LiveCodeBench evaluation (code_generation_lite)."""

    def __init__(self, rows: List[Mapping[str, Any]]):
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[LiveCodeBenchInstance]:
        for row in self._rows:
            yield LiveCodeBenchInstance(
                instance_id=str(row.get("question_id")),
                question_id=str(row.get("question_id")),
                question_content=str(row.get("question_content", "")),
                starter_code=str(row.get("starter_code", "") or ""),
                difficulty=str(row.get("difficulty", "unknown")),
                platform=str(row.get("platform", "unknown")),
                testtype=str(row.get("testtype", "stdin")),
                contest_date=str(row.get("contest_date", "") or ""),
                release=str(row.get("release", "") or ""),
                eval_sample=row["eval_sample"],
                metadata=(row.get("metadata") or {}),
            )

    @property
    def schema(self) -> Type[LiveCodeBenchInstance]:
        return LiveCodeBenchInstance

    @classmethod
    def _build(cls, rows, start_date, end_date, difficulties, testtypes,
               split, test_frac, seed, max_instances):
        rows = _filter_rows(rows, start_date, end_date, difficulties, testtypes)
        rows = _stratified_split(rows, split=split, test_frac=test_frac, seed=seed)
        if isinstance(max_instances, int) and max_instances >= 0:
            rows = rows[:max_instances]
        return cls(rows)

    @classmethod
    def from_hf(
        cls,
        release: str = "release_v6",
        start_date: Optional[str] = "2024-01-01",
        end_date: Optional[str] = None,
        difficulties: Optional[Sequence[str]] = None,
        testtypes: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        max_instances: Optional[int] = None,
        max_tests_per_problem: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> "LiveCodeBenchDataset":
        """Load+decode ``livecodebench/code_generation_lite`` (needs internet / HF cache).

        Defaults to the FULL window (``split=None``) to match the official LCB
        protocol; ``contest_date`` is filtered to ``>= start_date`` (default
        2024-01-01, after the Llama-3.x training cutoffs). Pass ``split='test'`` for
        the held-out stratified split (train/eval use only)."""
        rows = list(iter_release_rows(release, max_tests=max_tests_per_problem, cache_dir=cache_dir))
        return cls._build(rows, start_date, end_date, difficulties, testtypes,
                          split, test_frac, seed, max_instances)

    @classmethod
    def from_jsonl(
        cls,
        path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        difficulties: Optional[Sequence[str]] = None,
        testtypes: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        max_instances: Optional[int] = None,
    ) -> "LiveCodeBenchDataset":
        """Load a prebuilt snapshot JSONL (one built row per line; offline-friendly)."""
        p = Path(path)
        rows = [json.loads(line) for line in p.open() if line.strip()]
        return cls._build(rows, start_date, end_date, difficulties, testtypes,
                          split, test_frac, seed, max_instances)


################
#  Evaluator   #
################


class LiveCodeBenchEvaluator(Evaluator[LiveCodeBenchInstance]):
    """Scores a generation by running its extracted code against the problem's test
    cases (strict 0/1). The harness forks a child per problem (the vendored official
    LCB ``run_test`` mutates the interpreter + uses ``signal.alarm``)."""

    def __init__(self, timeout_seconds: float = 6.0, max_log_chars: int = 4000):
        self.timeout_seconds = float(timeout_seconds)
        self.max_log_chars = int(max_log_chars)

    def evaluate_sample(self, instance: LiveCodeBenchInstance, response: str) -> EvaluationResult:
        code = extract_code(response)
        if not code:
            return EvaluationResult(score=0.0, desc="empty code", metadata={"question_id": instance.question_id})
        ok = passed_all(instance.eval_sample, code, timeout=self.timeout_seconds)
        desc = code if len(code) <= self.max_log_chars else code[: self.max_log_chars] + "\n...[truncated]"
        return EvaluationResult(
            score=1.0 if ok else 0.0,
            desc=desc,
            metadata={"question_id": instance.question_id,
                      "difficulty": instance.difficulty,
                      "testtype": instance.testtype},
        )


##########################
# Prompt formatter (LM)  #
##########################


def default_prompt_formatter(tokenizer, instance: LiveCodeBenchInstance,
                             use_chat_format: bool = False) -> List[int]:
    """Build the LCB prompt for ``instance`` and return token ids.

    ``use_chat_format=True`` applies the model chat template (for instruct models);
    otherwise a plain system+body completion prompt (for base models)."""
    row = {"question_content": instance.question_content, "starter_code": instance.starter_code}
    text = format_lcb_prompt(row, tokenizer=tokenizer, chat_template=use_chat_format)
    return tokenizer.encode(text)
