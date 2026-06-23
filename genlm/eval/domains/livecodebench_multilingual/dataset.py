"""Dataset for multilingual LiveCodeBench (stdin/stdout problems only).

Reuses the existing LiveCodeBench loader/filters and forces ``testtypes=["stdin"]`` (no
functional/LeetCode problems, no conversion). Each problem is tagged with a target
``language``; one dataset is built per language. The raw ``question_id`` is preserved as its
own field so cross-language grouping survives the composite ``instance_id`` (``<qid>@<lang>``).
"""

from __future__ import annotations

from typing import Any, Iterator, List, Mapping, Optional, Sequence, Type

from genlm.eval.core import Dataset
from genlm.eval.domains.livecodebench.livecodebench import (
    LiveCodeBenchDataset,
    LiveCodeBenchInstance,
)

from .languages import resolve_language

_STDIN_ONLY = ("stdin",)


class MultilingualLCBInstance(LiveCodeBenchInstance):
    """One LiveCodeBench stdin problem paired with a target language.

    ``instance_id`` is the composite ``<question_id>@<language>`` (so the runner caches each
    language separately); ``question_id`` keeps the raw id for grouping/metadata.
    """

    language: str
    question_id: str


class MultilingualLCBDataset(Dataset[MultilingualLCBInstance]):
    """LiveCodeBench stdin problems for a single target language."""

    def __init__(self, rows: List[Mapping[str, Any]], language: str):
        self.language = resolve_language(language).key
        self._rows = list(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[MultilingualLCBInstance]:
        for row in self._rows:
            if row.get("question_id") is None:
                raise ValueError("LiveCodeBench row is missing question_id")
            qid = str(row["question_id"])
            yield MultilingualLCBInstance(
                instance_id=f"{qid}@{self.language}",
                question_id=qid,
                language=self.language,
                question_content=row.get("question_content", ""),
                starter_code=row.get("starter_code") or "",
                difficulty=row.get("difficulty", "unknown"),
                platform=row.get("platform", "unknown"),
                testtype=row.get("testtype", "stdin"),
                contest_date=row.get("contest_date") or "",
                eval_sample=row.get("eval_sample") or {},
            )

    @property
    def schema(self) -> Type[MultilingualLCBInstance]:
        return MultilingualLCBInstance

    def to_jsonl(self, path) -> None:
        """Write the (stdin-only) rows as a language-independent snapshot.

        ``language`` is a per-run tag and is not stored, so one snapshot serves all
        languages; reload it with ``from_jsonl(path, language=...)``.
        """
        # Delegate to the base writer over the same rows.
        LiveCodeBenchDataset(self._rows).to_jsonl(path)

    @classmethod
    def from_hf(
        cls,
        language: str,
        *,
        release: str = "release_v6",
        start_date: Optional[str] = "2024-01-01",
        end_date: Optional[str] = None,
        difficulties: Optional[Sequence[str]] = None,
        holdout: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        shuffle: bool = False,
        max_instances: Optional[int] = None,
        max_tests_per_problem: Optional[int] = None,
        cumulative: bool = True,
        cache_dir: Optional[str] = None,
    ) -> "MultilingualLCBDataset":
        """Load stdin LiveCodeBench problems for ``language`` (testtypes forced to stdin).

        The default ``start_date='2024-01-01'`` (inherited from the base loader) restricts the
        contest window. To reproduce Multi-LCB's problem set exactly, pass the release/window
        they used (e.g. ``start_date=None`` for the full ``release`` set, or their
        contamination cutoff like ``start_date='2024-07-01'``).
        """
        resolve_language(language)  # validate early
        base = LiveCodeBenchDataset.from_hf(
            release=release,
            start_date=start_date,
            end_date=end_date,
            difficulties=difficulties,
            testtypes=_STDIN_ONLY,
            holdout=holdout,
            test_frac=test_frac,
            seed=seed,
            shuffle=shuffle,
            max_instances=max_instances,
            max_tests_per_problem=max_tests_per_problem,
            cumulative=cumulative,
            cache_dir=cache_dir,
        )
        return cls(base._rows, language)

    @classmethod
    def from_jsonl(
        cls,
        path,
        language: str,
        *,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        difficulties: Optional[Sequence[str]] = None,
        holdout: Optional[str] = None,
        test_frac: float = 0.3,
        seed: int = 12345,
        shuffle: bool = False,
        max_instances: Optional[int] = None,
    ) -> "MultilingualLCBDataset":
        """Load stdin problems from a snapshot JSONL for ``language`` (testtypes forced)."""
        resolve_language(language)
        base = LiveCodeBenchDataset.from_jsonl(
            path,
            start_date=start_date,
            end_date=end_date,
            difficulties=difficulties,
            testtypes=_STDIN_ONLY,
            holdout=holdout,
            test_frac=test_frac,
            seed=seed,
            shuffle=shuffle,
            max_instances=max_instances,
        )
        return cls(base._rows, language)
