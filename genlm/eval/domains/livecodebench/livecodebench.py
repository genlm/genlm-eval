from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

from genlm.eval.core import Dataset, EvaluationResult, Evaluator, Instance

from genlm.eval.domains.livecodebench.harness import passed_all
from genlm.eval.domains.livecodebench.fetch import iter_release_rows
from genlm.eval.domains.livecodebench.prompts import RAW_STYLES, extract_code, format_lcb_prompt


class LiveCodeBenchInstance(Instance):
    """Schema for one LiveCodeBench problem (``instance_id`` is the question_id).

    ``eval_sample`` is the harness-ready ``{"input_output": <json str>}`` payload
    (decoded test cases); ``testtype`` is ``stdin`` or ``functional``.
    """

    question_content: str
    starter_code: str = ""
    difficulty: str = "unknown"
    platform: str = "unknown"
    testtype: str = "stdin"
    contest_date: str = ""
    # Pydantic v2 deep-copies this {} default per instance (not shared state).
    eval_sample: Dict[str, str] = {}  # may be empty for a prompts-only (generation) snapshot


def _holdout_split(rows: List[dict], holdout: Optional[str], test_frac: float, seed: int) -> List[dict]:
    """Stratified (testtype, difficulty) holdout; all rows when holdout=None.

    Deterministic given a fixed seed and stable upstream row order."""
    if holdout is None:
        return rows
    if holdout not in ("train", "test"):
        raise ValueError(f"holdout must be 'train', 'test', or None; got {holdout!r}")
    by_key: Dict[Any, list] = {}
    for r in rows:
        by_key.setdefault((r.get("testtype", "stdin"), r.get("difficulty", "unknown")), []).append(r)
    rng = random.Random(seed)
    out: List[dict] = []
    for key in sorted(by_key):
        items = list(by_key[key])
        rng.shuffle(items)
        n_test = int(round(len(items) * test_frac))
        out.extend(items[:n_test] if holdout == "test" else items[n_test:])
    rng.shuffle(out)
    return out


def _parse_window(start_date: Optional[str], end_date: Optional[str]
                  ) -> Tuple[Optional[datetime], Optional[datetime]]:
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    return start_dt, end_dt


def _in_window(contest_date: Any, start_dt: Optional[datetime], end_dt: Optional[datetime]) -> bool:
    """Match official lcb_runner: compare contest_date as a datetime against the window
    bounds (YYYY-MM-DD = midnight), inclusive on both ends. So a timed contest on the
    end_date day (e.g. ...T19:30:00) is EXCLUDED, as upstream. Undated/unparseable
    rows are dropped whenever a window is set."""
    if start_dt is None and end_dt is None:
        return True
    try:
        cd = datetime.fromisoformat(str(contest_date or ""))
    except ValueError:
        return False
    if cd.tzinfo is not None:  # window bounds are naive; drop the offset to compare
        cd = cd.replace(tzinfo=None)
    if start_dt and cd < start_dt:
        return False
    if end_dt and cd > end_dt:
        return False
    return True


def _filter_rows(rows: Iterable[dict], start_date: Optional[str], end_date: Optional[str],
                 difficulties: Optional[Sequence[str]], testtypes: Optional[Sequence[str]]) -> List[dict]:
    diff_set = {d.lower() for d in difficulties} if difficulties else None
    tt_set = {t.lower() for t in testtypes} if testtypes else None
    start_dt, end_dt = _parse_window(start_date, end_date)
    out = []
    for r in rows:
        if not _in_window(r.get("contest_date"), start_dt, end_dt):
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
                instance_id=row["question_id"],
                question_content=row.get("question_content", ""),
                starter_code=row.get("starter_code") or "",
                difficulty=row.get("difficulty", "unknown"),
                platform=row.get("platform", "unknown"),
                testtype=row.get("testtype", "stdin"),
                contest_date=row.get("contest_date") or "",
                eval_sample=row.get("eval_sample") or {},
            )

    @property
    def schema(self) -> Type[LiveCodeBenchInstance]:
        return LiveCodeBenchInstance

    def to_jsonl(self, path) -> None:
        """Write the (already filtered/split) rows as a snapshot JSONL.

        The recommended way to build offline snapshots: load with ``from_hf``,
        then ``to_jsonl`` — so the snapshot inherits the date window and reloading
        it with ``from_jsonl`` (which applies no window by default) matches."""
        with Path(path).open("w") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")

    @classmethod
    def _build(cls, rows, start_date, end_date, difficulties, testtypes,
               holdout, test_frac, seed, shuffle, max_instances):
        rows = _filter_rows(rows, start_date, end_date, difficulties, testtypes)
        rows = _holdout_split(rows, holdout=holdout, test_frac=test_frac, seed=seed)
        if shuffle:
            random.Random(seed).shuffle(rows)
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
        holdout: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        shuffle: bool = False,
        max_instances: Optional[int] = None,
        max_tests_per_problem: Optional[int] = None,
        cumulative: bool = True,
        cache_dir: Optional[str] = None,
    ) -> "LiveCodeBenchDataset":
        """Load+decode ``livecodebench/code_generation_lite`` (needs internet / HF cache).

        ``cumulative=True`` = official version_tag semantics (all problems through
        ``release``). Defaults to the full benchmark (``holdout=None``) with
        ``contest_date >= start_date`` (2024-01-01 = after the Llama-3.x cutoffs).
        ``holdout='train'``/``'test'`` selects a stratified random partition (NOT the
        HF split name) — leave it None for leaderboard-comparable numbers."""
        start_dt, end_dt = _parse_window(start_date, end_date)
        rows = list(iter_release_rows(
            release, max_tests=max_tests_per_problem, cache_dir=cache_dir,
            cumulative=cumulative,
            # prefilter on the raw row so out-of-window private tests are never decoded
            raw_filter=lambda raw: _in_window(raw.get("contest_date"), start_dt, end_dt),
        ))
        return cls._build(rows, start_date, end_date, difficulties, testtypes,
                          holdout, test_frac, seed, shuffle, max_instances)

    @classmethod
    def from_jsonl(
        cls,
        path,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        difficulties: Optional[Sequence[str]] = None,
        testtypes: Optional[Sequence[str]] = None,
        holdout: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        shuffle: bool = False,
        max_instances: Optional[int] = None,
    ) -> "LiveCodeBenchDataset":
        """Load a snapshot JSONL written by ``to_jsonl`` (offline-friendly).

        Unlike ``from_hf``, applies NO date window by default: the snapshot is
        taken as-is (it already carries the window it was built with)."""
        with Path(path).open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return cls._build(rows, start_date, end_date, difficulties, testtypes,
                          holdout, test_frac, seed, shuffle, max_instances)


class LiveCodeBenchEvaluator(Evaluator[LiveCodeBenchInstance]):
    """Runs a generation's extracted code against the problem's test cases (strict 0/1).

    Results are memoized on (instance_id, extracted code) — under particle-based
    inference many responses are byte-identical and the harness is deterministic.
    ``max_total_seconds`` (optional) caps the per-sample wall-clock budget; see
    ``check_correctness``."""

    def __init__(self, timeout_seconds: float = 6.0, max_log_chars: int = 4000,
                 max_total_seconds: Optional[float] = None, extraction_style: str = "generic"):
        self.timeout_seconds = float(timeout_seconds)
        self.max_log_chars = int(max_log_chars)
        self.max_total_seconds = max_total_seconds
        self.extraction_style = extraction_style  # "genericbase" for base-model generations
        self._cache: Dict[Tuple[Any, str], bool] = {}

    def _passed(self, instance: LiveCodeBenchInstance, code: str) -> bool:
        key = (instance.instance_id, code)
        if key not in self._cache:
            self._cache[key] = passed_all(instance.eval_sample, code,
                                          timeout=self.timeout_seconds,
                                          max_total_seconds=self.max_total_seconds)
        return self._cache[key]

    def evaluate_sample(self, instance: LiveCodeBenchInstance, response: str) -> EvaluationResult:
        code = extract_code(response, style=self.extraction_style)
        if not code:
            return EvaluationResult(score=0.0, desc="empty code",
                                    metadata={"question_id": instance.instance_id})
        if not instance.eval_sample or "input_output" not in instance.eval_sample:
            return EvaluationResult(score=0.0, desc="missing eval_sample",
                                    metadata={"question_id": instance.instance_id})
        ok = self._passed(instance, code)
        desc = code if len(code) <= self.max_log_chars else code[: self.max_log_chars] + "\n...[truncated]"
        return EvaluationResult(
            score=1.0 if ok else 0.0,
            desc=desc,
            metadata={"question_id": instance.instance_id,
                      "difficulty": instance.difficulty,
                      "testtype": instance.testtype},
        )


def default_prompt_formatter(tokenizer, instance: LiveCodeBenchInstance,
                             use_chat_format: bool = False, style: str = "generic") -> List[int]:
    """Build the LCB prompt for ``instance`` and return token ids.

    style="generic" + use_chat_format=True = LLaMa3 lcb_runner style (chat template).
    style="codeqwen"/"deepseek"/"genericbase" = raw completion strings; genericbase
    needs the matching evaluator extraction_style."""
    row = {"question_content": instance.question_content, "starter_code": instance.starter_code}
    text = format_lcb_prompt(row, tokenizer=tokenizer, chat_template=use_chat_format, style=style)
    if style in RAW_STYLES:
        return tokenizer.encode(text)  # raw completion string; vLLM-default specials
    # Chat template already includes the BOS; avoid a second one on re-encode.
    return tokenizer.encode(text, add_special_tokens=not use_chat_format)
