"""Ag-MBPP-X (nuprl/mbpp-agnostic-translation) as a multilingual stdin/stdout eval set.

The Agnostics group rewrote MBPP's assert-based tasks into language-agnostic problems with
stdin/stdout I/O tests. That makes them a drop-in OUT-OF-DOMAIN eval for the multilingual-LCB
harness: MBPP upstream, so zero Codeforces overlap with the LCB split.

The rows are LLM-generated, so this loader treats the dataset as UNTRUSTED input and validates
hard before anything reaches a prompt or an executor:

  * the HF revision is pinned; a moved dataset cannot silently change the eval
  * schema, size and character checks on every field (control characters, oversized tests,
    empty I/O), with a scan for prompt-injection markers in the text fields
  * contradictory tests (same input, different expected output) drop the whole problem: for
    LLM-generated tests this is the signature of a broken row, and one bad hidden test would
    silently misgrade every model on that problem
  * exact and normalized-text dedup, so repeated problems cannot double-weight the metric

Validation is drop-and-count by default (``strict=True`` raises instead); the per-reason drop
counts are kept on the dataset so a run can assert nothing unexpected was lost.

The first test is embedded in the statement as the worked example, which makes it the PUBLIC
test both for ``public_eval_sample`` consumers and for verbatim-in-prompt detectors; the rest
stay hidden. Problems with fewer than 2 tests are dropped so at least one hidden test always
remains.
"""

import json
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Type

from .dataset import Dataset, MultilingualLCBInstance, resolve_language

HF_REPO = "nuprl/mbpp-agnostic-translation"
# a32a76f = 2026-03-13, the revision the validation numbers below were measured on
PINNED_REVISION = "a32a76fa8a7ae4b6b2eeee31e086d1b33635940c"

MAX_TEXT_CHARS = 20_000       # description + format prose
MAX_IO_CHARS = 65_536         # any single test input or output
MAX_TESTS = 200

# Narrow, high-precision markers. These do not occur in legitimate MBPP-style problem prose;
# anything matching is treated as a poisoned row, not a false positive to argue with.
_INJECTION = re.compile(
    r"(ignore (all )?(previous|prior) (instructions|messages)"
    r"|disregard (the|all|your) (system|previous|prior)"
    r"|<\s*/?\s*system\b"
    r"|\bBEGIN SYSTEM PROMPT\b"
    r"|\x1b\[)",
    re.IGNORECASE,
)
# printable text plus ordinary whitespace; everything else is suspect in I/O strings
_BAD_CHARS = re.compile(r"[^\x20-\x7e\t\n\r -￿]")


def _norm(text: str) -> str:
    """Normalization for dedup: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _check_text(field: str, value: Any, drops: List[str]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        drops.append(f"{field}:empty")
        return None
    if len(value) > MAX_TEXT_CHARS:
        drops.append(f"{field}:oversize")
        return None
    if _INJECTION.search(value):
        drops.append(f"{field}:injection_marker")
        return None
    if _BAD_CHARS.search(value):
        drops.append(f"{field}:control_chars")
        return None
    return value


def _check_tests(raw: Any, drops: List[str]) -> Optional[List[Tuple[str, str]]]:
    if not isinstance(raw, (list, tuple)) or not (2 <= len(raw) <= MAX_TESTS):
        drops.append("tests:count")
        return None
    seen: Dict[str, str] = {}
    tests: List[Tuple[str, str]] = []
    for t in raw:
        if not isinstance(t, Mapping):
            drops.append("tests:shape")
            return None
        inp, out = t.get("input"), t.get("output")
        if not isinstance(inp, str) or not isinstance(out, str) or not inp.strip() or out == "":
            drops.append("tests:empty_io")
            return None
        if len(inp) > MAX_IO_CHARS or len(out) > MAX_IO_CHARS:
            drops.append("tests:oversize_io")
            return None
        if _BAD_CHARS.search(inp) or _BAD_CHARS.search(out):
            drops.append("tests:control_chars")
            return None
        key = inp.strip()
        prev = seen.get(key)
        if prev is not None:
            if prev.strip() != out.strip():
                # contradictory expectations for the same input: the row is broken and any
                # grading against it is noise at best
                drops.append("tests:contradictory")
                return None
            continue  # exact duplicate test: keep one copy
        seen[key] = out
        tests.append((inp, out))
    if len(tests) < 2:
        drops.append("tests:count")
        return None
    return tests


def _statement(desc: str, in_fmt: str, out_fmt: str, example: Tuple[str, str]) -> str:
    return (
        f"{desc.strip()}\n\n"
        f"Input Format:\n{in_fmt.strip()}\n\n"
        f"Output Format:\n{out_fmt.strip()}\n\n"
        f"Example\n\nInput\n\n{example[0].strip()}\n\nOutput\n\n{example[1].strip()}\n"
    )


def _io_blob(tests: List[Tuple[str, str]]) -> Dict[str, str]:
    return {"input_output": json.dumps(
        {"inputs": [t[0] for t in tests], "outputs": [t[1] for t in tests]})}


class MBPPAgnosticDataset(Dataset[MultilingualLCBInstance]):
    """Ag-MBPP-X problems for one target language, validated and deduplicated.

    Instances carry ``platform='mbpp-agnostic'`` and ``question_id='mbppx_<task_id>'`` so
    nothing downstream can confuse them with LCB problems or leak them into an LCB split.
    """

    def __init__(self, rows: List[Mapping[str, Any]], language: str,
                 drop_counts: Optional[Dict[str, int]] = None):
        self.language = resolve_language(language).key
        self._rows = list(rows)
        self.drop_counts = dict(drop_counts or {})

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[MultilingualLCBInstance]:
        for row in self._rows:
            qid = row["question_id"]
            yield MultilingualLCBInstance(
                instance_id=f"{qid}@{self.language}",
                question_id=qid,
                language=self.language,
                question_content=row["question_content"],
                starter_code="",
                difficulty="unknown",
                platform="mbpp-agnostic",
                testtype="stdin",
                contest_date="",
                eval_sample=row["eval_sample"],
                public_eval_sample=row["public_eval_sample"],
            )

    @property
    def schema(self) -> Type[MultilingualLCBInstance]:
        return MultilingualLCBInstance

    @classmethod
    def from_rows(cls, raw_rows: List[Mapping[str, Any]], language: str,
                  *, strict: bool = False) -> "MBPPAgnosticDataset":
        """Validate raw dataset rows; see the module docstring for the checks applied."""
        rows: List[Dict[str, Any]] = []
        drop_counts: Dict[str, int] = {}
        seen_ids: set = set()
        seen_norm: set = set()
        for raw in raw_rows:
            drops: List[str] = []
            task_id = raw.get("original_task_id")
            if not isinstance(task_id, (int, str)) or str(task_id).strip() == "":
                drops.append("task_id:missing")
            desc = _check_text("description", raw.get("description"), drops)
            in_fmt = _check_text("input_format", raw.get("input_format"), drops)
            out_fmt = _check_text("output_format", raw.get("output_format"), drops)
            tests = _check_tests(raw.get("tests"), drops) if not drops else None
            if not drops:
                qid = f"mbppx_{task_id}"
                norm = _norm(desc)
                if qid in seen_ids:
                    drops.append("dedup:task_id")
                elif norm in seen_norm:
                    drops.append("dedup:description")
                else:
                    seen_ids.add(qid)
                    seen_norm.add(norm)
            if drops:
                if strict:
                    raise ValueError(f"row {raw.get('original_task_id')!r} failed: {drops}")
                for d in drops:
                    drop_counts[d] = drop_counts.get(d, 0) + 1
                continue
            public, hidden = tests[:1], tests[1:]
            rows.append({
                "question_id": qid,
                "question_content": _statement(desc, in_fmt, out_fmt, public[0]),
                "eval_sample": _io_blob(hidden),
                "public_eval_sample": _io_blob(public),
            })
        return cls(rows, language, drop_counts)

    @classmethod
    def from_hf(cls, language: str, *, config: str = "sanitized",
                revision: str = PINNED_REVISION, strict: bool = False,
                cache_dir: Optional[str] = None) -> "MBPPAgnosticDataset":
        """Load from HF at the PINNED revision (pass ``revision`` explicitly to move it)."""
        from datasets import load_dataset

        ds = load_dataset(HF_REPO, config, revision=revision, cache_dir=cache_dir)
        raw = list(ds[next(iter(ds.keys()))])
        return cls.from_rows(raw, language, strict=strict)

    def overlap_with(self, other: "Dataset") -> List[Tuple[str, str]]:
        """Normalized-description collisions against another dataset (contamination check).

        Returns [(our question_id, their question_id)]; expected empty against LCB.
        """
        theirs = {_norm(i.question_content): i.question_id for i in other}
        hits = []
        for row in self._rows:
            n = _norm(row["question_content"])
            if n in theirs:
                hits.append((row["question_id"], theirs[n]))
        return hits
