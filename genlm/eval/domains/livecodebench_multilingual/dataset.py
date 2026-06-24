"""Dataset and language registry for multilingual LiveCodeBench (stdin/stdout problems only).

Reuses the existing LiveCodeBench loader and forces ``testtypes=["stdin"]``. The raw
``question_id`` is kept as its own field so cross-language grouping survives the composite
``instance_id`` (``<qid>@<lang>``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Type

from genlm.eval.core import Dataset
from genlm.eval.domains.livecodebench.livecodebench import (
    LiveCodeBenchDataset,
    LiveCodeBenchInstance,
)

_STDIN_ONLY = ("stdin",)


@dataclass(frozen=True)
class Language:
    key: str  # canonical eval_scripts key, e.g. "c++"
    display: str  # name shown in the system message, e.g. "C++"
    md_fence: str  # markdown code-fence tag, e.g. "cpp"
    comment: str  # single-line comment token, e.g. "//"
    tier: int  # 1 Multi-LCB, 2 low-resource, 3 high-risk
    source: str  # "multilcb" | "agnostics"
    prompt_nudge: str = ""  # appended guidance (low-resource langs only)


# 12 Multi-LCB languages; display/md_fence/comment mirror lcb_runner.utils.PLang so prompts are
# byte-identical to Multi-LCB (php's display is lowercase upstream, kept for parity).
_MULTILCB = [
    Language("python", "Python", "python", "#", 1, "multilcb"),
    Language("c++", "C++", "cpp", "//", 1, "multilcb"),
    Language("java", "Java", "java", "//", 1, "multilcb"),
    Language("c#", "C#", "csharp", "//", 1, "multilcb"),
    Language("go", "Go", "go", "//", 1, "multilcb"),
    Language("javascript", "JavaScript", "javascript", "//", 1, "multilcb"),
    Language("typescript", "TypeScript", "typescript", "//", 1, "multilcb"),
    Language("rust", "Rust", "rust", "//", 1, "multilcb"),
    Language("ruby", "Ruby", "ruby", "#", 1, "multilcb"),
    Language("php", "php", "php", "//", 1, "multilcb"),
    Language("kotlin", "Kotlin", "kotlin", "//", 1, "multilcb"),
    Language("scala", "Scala", "scala", "//", 1, "multilcb"),
]

# 5 Agnostics low-resource languages (nudges paraphrased; Agnostics ships no license).
_AGNOSTICS = [
    Language(
        "lua",
        "Lua",
        "lua",
        "--",
        2,
        "agnostics",
        prompt_nudge="Target Lua 5.1 / LuaJIT.",
    ),
    Language(
        "julia",
        "Julia",
        "julia",
        "#",
        2,
        "agnostics",
        prompt_nudge="Target Julia 1.11.",
    ),
    Language(
        "r",
        "R",
        "r",
        "#",
        2,
        "agnostics",
        prompt_nudge=(
            'Target R 4. Read stdin with readLines(con = file("stdin")) (the optional n '
            "argument limits how many lines are read) and write output with cat; do not use "
            "print."
        ),
    ),
    Language(
        "ocaml",
        "OCaml",
        "ocaml",
        "(*",
        3,
        "agnostics",
        prompt_nudge=(
            "Target OCaml 5 using the standard library for I/O (Scanf/Printf, read_line). "
            "Remember the dotted float operators (+. -. *. /.), explicit int/float casts, "
            "and that lists favour pattern matching or folds over indexing."
        ),
    ),
    Language(
        "fortran",
        "Fortran",
        "fortran",
        "!",
        2,
        "agnostics",
        prompt_nudge=(
            "Target Fortran 90. Begin each scope with implicit none; arrays are 1-based; "
            "read a size before allocating and reading an array; use real literals (e.g. "
            "2.0d0) to avoid integer division; read inputs, compute, and write output only."
        ),
    ),
]

LANGUAGES: Dict[str, Language] = {lang.key: lang for lang in (_MULTILCB + _AGNOSTICS)}

# Aliases accepted by resolve_language (canonical keys also resolve to themselves).
_ALIASES = {
    "cpp": "c++",
    "cplusplus": "c++",
    "csharp": "c#",
    "cs": "c#",
    "js": "javascript",
    "ts": "typescript",
    "golang": "go",
}


def resolve_language(name: str) -> "Language":
    """Resolve a language name (case-insensitive, with aliases) to a Language; raises ValueError."""
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in LANGUAGES:
        raise ValueError(
            f"unknown language {name!r}; known: {sorted(LANGUAGES)} (aliases: {sorted(_ALIASES)})"
        )
    return LANGUAGES[key]


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
        """Write the stdin rows as a language-independent snapshot (language is a per-run tag,
        not stored; reload with ``from_jsonl(path, language=...)``)."""
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

        ``start_date`` defaults to the base loader's ``2024-01-01``; pass the paper's window
        (e.g. ``None`` or ``'2024-07-01'``) to match a specific problem set.
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
