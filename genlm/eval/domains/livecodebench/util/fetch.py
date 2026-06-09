"""Load + decode LiveCodeBench ``code_generation_lite`` releases.

Downloads a release's ``testN.jsonl`` via ``huggingface_hub`` (the ``datasets``
builder hits a pyarrow offset-overflow on the large ``private_test_cases`` column),
decodes the private tests, and emits a harness-ready ``eval_sample`` per problem.
``iter_release_rows`` feeds ``LiveCodeBenchDataset.from_hf``.
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any, Dict, Iterator, List, Mapping, Optional

from huggingface_hub import hf_hub_download

# HF Lite variant: https://huggingface.co/datasets/livecodebench/code_generation_lite
HF_REPO = "livecodebench/code_generation_lite"


def _decode_private(field: str) -> List[Dict[str, Any]]:
    """Plain JSON, else base64 -> zlib -> pickle -> json (LCB chain).

    The pickle branch trusts the dataset publisher (as official lcb_runner does)."""
    try:
        return json.loads(field)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(field.encode("utf-8")))))


def derive_testtype(metadata: Mapping[str, Any], tests: List[Mapping[str, Any]]) -> str:
    """functional iff a func_name is present (matches run_test's which_type),
    else fall back to the first test's recorded testtype, else stdin."""
    if metadata.get("func_name"):
        return "functional"
    if tests and tests[0].get("testtype"):
        return str(tests[0]["testtype"])
    return "stdin"


def build_row(raw: Mapping[str, Any], release: str, max_tests: Optional[int] = None) -> Dict[str, Any]:
    """Convert a raw HF row into a clean snapshot row with a harness-ready
    ``eval_sample`` (``{"input_output": <json str>}``)."""
    public = json.loads(raw["public_test_cases"])
    private = _decode_private(raw["private_test_cases"])
    all_tests = list(public) + list(private)
    metadata = json.loads(raw["metadata"]) if isinstance(raw.get("metadata"), str) else (raw.get("metadata") or {})

    inputs = [t["input"] for t in all_tests]
    outputs = [t["output"] for t in all_tests]
    if max_tests is not None:
        inputs, outputs = inputs[:max_tests], outputs[:max_tests]

    eval_sample = {"input_output": json.dumps({
        "inputs": inputs, "outputs": outputs, "fn_name": metadata.get("func_name"),
    })}
    return {
        "question_id": raw.get("question_id"),
        "question_content": raw.get("question_content", ""),
        "starter_code": raw.get("starter_code", "") or "",
        "difficulty": raw.get("difficulty", "unknown"),
        "platform": raw.get("platform", "unknown"),
        "contest_date": raw.get("contest_date", ""),
        "testtype": derive_testtype(metadata, all_tests),
        "release": release,
        "eval_sample": eval_sample,
        "metadata": metadata,
    }


def _release_num(release: str) -> int:
    """``release_vN`` -> ``N``; raises on anything but that shape."""
    if "_v" not in release:
        raise ValueError(f"release must be 'release_vN'; got {release!r}")
    try:
        return int(release.rsplit("_v", 1)[1])
    except ValueError:
        raise ValueError(f"release must be 'release_vN'; got {release!r}")


def _release_filename(release: str) -> str:
    """``release_vN`` -> ``testN.jsonl`` (``release_v1`` -> ``test.jsonl``)."""
    n = _release_num(release)
    return "test.jsonl" if n == 1 else f"test{n}.jsonl"


def iter_release_rows(release: str = "release_v6", max_tests: Optional[int] = None,
                      cache_dir: Optional[str] = None, cumulative: bool = True
                      ) -> Iterator[Dict[str, Any]]:
    """Yield clean built rows for a release (needs HF cache).

    ``cumulative=True`` (official version_tag semantics) loads test.jsonl..testN.jsonl
    de-duped by question_id (release_v6 == ~1055); ``cumulative=False`` loads only that
    window. Dedup keeps the first occurrence, so ``release`` = first-seen."""
    n = _release_num(release)
    tags = [f"release_v{i}" for i in range(1, n + 1)] if cumulative else [release]
    seen = set()
    for tag in tags:
        path = hf_hub_download(repo_id=HF_REPO, filename=_release_filename(tag),
                               repo_type="dataset", cache_dir=cache_dir)
        with open(path) as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = build_row(json.loads(line), release=tag, max_tests=max_tests)
                qid = row["question_id"]
                if qid in seen:
                    continue
                seen.add(qid)
                yield row
