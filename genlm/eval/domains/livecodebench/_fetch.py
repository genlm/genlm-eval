"""Load + decode LiveCodeBench ``code_generation_lite`` releases.

Ports the snapshot-builder logic from the genlm/latent PR
(``data/livecodebench/fetch.py``): download one release's ``testN.jsonl`` via
``huggingface_hub`` (bypassing the ``datasets`` builder and its pyarrow
offset-overflow on the large ``private_test_cases`` column), decode the private
tests, and emit a clean row with a harness-ready ``eval_sample`` per problem.

``iter_release_rows`` is what ``LiveCodeBenchDataset.from_hf`` consumes; the
``build_row`` / ``derive_testtype`` / ``_decode_private`` helpers are kept public
so the unit tests can exercise the decode chain directly.
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from typing import Any, Dict, Iterator, List, Mapping, Optional

# HF dataset: the *lite* variant (capped/compressed tests) — NOT ``code_generation``.
HF_REPO = "livecodebench/code_generation_lite"


def _decode_private(field: str) -> List[Dict[str, Any]]:
    """Plain JSON if possible, else base64 -> zlib -> pickle -> json (LCB chain)."""
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


def _release_filename(release: str) -> str:
    """``release_vN`` -> ``testN.jsonl`` (``release_v1`` -> ``test.jsonl``)."""
    n = int(release.rsplit("_v", 1)[1])
    return "test.jsonl" if n == 1 else f"test{n}.jsonl"


def iter_release_rows(release: str = "release_v6", max_tests: Optional[int] = None,
                      cache_dir: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Download one release of ``code_generation_lite`` and yield clean built rows.

    Requires internet / a warm HF cache (run on a login node on offline clusters).
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO, filename=_release_filename(release),
                           repo_type="dataset", cache_dir=cache_dir)
    with open(path) as fin:
        for line in fin:
            if not line.strip():
                continue
            yield build_row(json.loads(line), release=release, max_tests=max_tests)
