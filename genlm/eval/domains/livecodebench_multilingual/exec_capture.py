#!/usr/bin/env python
"""Build the full execution record for multilingual-LCB rollouts.

The multilingual-LCB analogue of the LCB / DS-1000 execution record: persist each test's
untruncated stdout and per-test verdict so a run's inputs and outputs stay recoverable without
re-executing untrusted code.

Two modes (CLI `--mode`):

  cell      Re-grade one (model, arm, temp) cell's already-banked completions with no short-circuit
            (capture.capture_run) and write the execution star schema:
              executions/domain=mlcb/language=<lang>/model=<tag>/temp=<t>/data.parquet
                  one row per (instance, sample, test_idx): verdict + hashes + error + time
              exec_outputs/domain=mlcb/language=<lang>/<cell>.parquet
                  each unique program stdout stored once by output_hash
            Generation-free: reads completion jsonl shards, so it backfills cells already on HF.

  problems  Build the static problem-definition tables from the snapshot (run once per language):
              problems/domain=mlcb/language=<lang>/data.parquet
                  one row per question_id: prompt + starter_code + difficulty + inputs/expected
              exec_inputs/domain=mlcb/language=<lang>/data.parquet
                  each unique test input / expected output once by hash (FK for executions)

The fact-table columns match genlm-rollouts `executions` so the table unions across domains.
Grading runs untrusted code: run on a sandboxed CPU host with the toolchain on PATH.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from .capture import capture_run
from .dataset import MultilingualLCBDataset
from .executor import is_toolchain_available
from .prompts import default_grading, extract_code
from .vendored.testing_plang import Status

EXEC_SCHEMA = pa.schema([
    ("model", pa.string()), ("temp", pa.string()),
    ("instance_id", pa.string()), ("sample", pa.int32()),
    ("test_idx", pa.int32()), ("n_tests", pa.int32()),
    ("passed", pa.bool_()), ("error_code", pa.int8()), ("output_kind", pa.string()),
    ("input_hash", pa.string()), ("expected_hash", pa.string()),
    ("output_hash", pa.string()), ("canonical_output_hash", pa.string()),
    ("error_hash", pa.string()), ("error_message", pa.string()), ("time_s", pa.float32()),
    ("artifact_sha256", pa.string()), ("expected_artifact_sha256", pa.string()),
    ("library", pa.string()), ("seed", pa.int32()), ("deterministic", pa.bool_()),
])
OUTPUT_SCHEMA = pa.schema([
    ("output_hash", pa.string()), ("output_kind", pa.string()),
    ("output", pa.string()), ("size", pa.int32()),
])
PROB_SCHEMA = pa.schema([
    ("question_id", pa.string()), ("instance_id", pa.string()),
    ("difficulty", pa.string()), ("platform", pa.string()),
    ("testtype", pa.string()), ("contest_date", pa.string()),
    ("question_content", pa.string()), ("starter_code", pa.string()),
    ("n_tests", pa.int32()),
    # ordered per-test hashes; resolve to text via exec_inputs (input_hash to input). Hashes (not
    # the full text) keep this readable: pyarrow cannot read large strings nested in list columns.
    ("input_hashes", pa.list_(pa.string())), ("expected_hashes", pa.list_(pa.string())),
])
INPUT_SCHEMA = pa.schema([
    ("input_hash", pa.string()), ("input_kind", pa.string()),
    ("input", pa.string()), ("size", pa.int32()),
])


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", "surrogatepass")).hexdigest()


def _load_tests(snapshot, language):
    """instance_id -> (inputs, outputs) and -> instance, from the snapshot."""
    insts = {i.instance_id: i for i in MultilingualLCBDataset.from_jsonl(snapshot, language)}
    tests = {}
    for iid, inst in insts.items():
        try:
            io = json.loads(inst.eval_sample["input_output"])
            tests[iid] = (io["inputs"], io["outputs"])
        except Exception:
            tests[iid] = ([], [])
    return insts, tests


# ---------------------------------------------------------------------------- cell capture
_TESTS: dict = {}
_LANG = ""
_GRADING = "exact"
_TIMEOUT = 10.0
_MAXCOMP = 1e9


def _init(tests, lang, grading, timeout, maxcomp):
    global _TESTS, _LANG, _GRADING, _TIMEOUT, _MAXCOMP
    _TESTS, _LANG, _GRADING, _TIMEOUT, _MAXCOMP = tests, lang, grading, float(timeout), float(maxcomp)


def _err_code(rec) -> int:
    if rec["passed"]:
        return 1
    if rec["status"] == "capped":
        return -1
    # capture._rec stores str(status), and str(Status.Done) == "Done". A cleanly-run failing test
    # (Done, not passed) is a wrong answer (-2); any other non-passing status is an exec failure.
    return -2 if rec["status"] == str(Status.Done) else -5


def _no_capture_records(n_tests):
    """Failing placeholder rows for a sample that never executed (empty or unparsed code), one per
    test so the row count matches executed samples. A zero-test instance yields a single row."""
    if n_tests <= 0:
        return [{"test_idx": 0, "passed": False, "output": "", "error_message": "no tests",
                 "error_code": -5, "status": "none", "time_s": 0.0}]
    return [{"test_idx": i, "passed": False, "output": "", "error_message": "empty or unparsed code",
             "error_code": -5, "status": "uncaptured", "time_s": 0.0} for i in range(n_tests)]


def _capture_one(arg):
    iid, code = arg
    io = _TESTS.get(iid)
    if not io or not io[0]:
        return (iid, code, [])
    inputs, outputs = io
    try:
        _, per = capture_run(code, inputs, outputs, _LANG, _TIMEOUT, _GRADING, _MAXCOMP)
    except Exception as e:
        per = [{"test_idx": i, "passed": False, "output": "", "error_message": f"capture error: {e}",
                "error_code": -5, "status": "capturefail", "time_s": 0.0} for i in range(len(outputs))]
    return (iid, code, per)


def _read_completions(patterns):
    recs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            with open(f) as fh:
                recs += [json.loads(line) for line in fh if line.strip()]
    for r in recs:
        if not (r.get("extracted_code") or "").strip():
            r["extracted_code"] = extract_code(r.get("answer_text") or "")
    return recs


def capture_cell(a):
    if not is_toolchain_available(a.language):
        raise SystemExit(f"toolchain for {a.language} not on PATH")
    grading = default_grading(a.language)
    recs = _read_completions(a.completions)
    if not recs:
        raise SystemExit(f"no completion rows matched {a.completions}")
    _, tests = _load_tests(a.snapshot, a.language)
    print(f"{a.cell}: {len(recs)} completions; grading={grading}", flush=True)

    uniq = sorted({(r["instance_id"], r["extracted_code"]) for r in recs if (r["extracted_code"] or "").strip()})
    print(f"  {len(uniq)} unique (instance, code) to capture", flush=True)
    per_code = {}
    if uniq:
        with ProcessPoolExecutor(a.workers, initializer=_init,
                                 initargs=(tests, a.language, grading, a.timeout, a.max_completion_seconds)) as ex:
            for iid, code, per in ex.map(_capture_one, uniq, chunksize=4):
                per_code[(iid, code)] = per
    print("  capture done; assembling tables", flush=True)

    exec_path = f"{a.out}/executions/domain=mlcb/language={a.language}/model={a.model}/temp={a.temp}/data.parquet"
    os.makedirs(os.path.dirname(exec_path), exist_ok=True)
    out_rows = {}
    buf = []
    writer = None
    n_exec = 0

    def _flush():
        nonlocal writer
        if not buf:
            return
        tbl = pa.Table.from_pylist(buf, schema=EXEC_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(exec_path, EXEC_SCHEMA, compression="zstd")
        writer.write_table(tbl)
        buf.clear()

    for r in recs:
        iid = r["instance_id"]
        code = r.get("extracted_code") or ""
        per = per_code.get((iid, code))
        sample = int(r.get("sample", 0))
        inputs, outputs = tests.get(iid) or ([], [])
        if not per:
            per = _no_capture_records(len(outputs))
        nt = len(per)
        for rec in per:
            ti = rec["test_idx"]
            oh = _sha(rec["output"])
            if oh not in out_rows:
                out_rows[oh] = rec["output"]
            inp = inputs[ti] if ti < len(inputs) else ""
            exp = outputs[ti] if ti < len(outputs) else ""
            buf.append({
                "model": a.model, "temp": str(a.temp), "instance_id": str(iid), "sample": sample,
                "test_idx": int(ti), "n_tests": nt, "passed": bool(rec["passed"]),
                "error_code": _err_code(rec), "output_kind": "stdio",
                "input_hash": _sha(inp), "expected_hash": _sha(exp),
                "output_hash": oh, "canonical_output_hash": oh,
                "error_hash": _sha(rec["error_message"]) if rec["error_message"] else "",
                "error_message": rec["error_message"] or "", "time_s": float(rec["time_s"]),
                "artifact_sha256": None, "expected_artifact_sha256": None,
                "library": None, "seed": None, "deterministic": True,
            })
            n_exec += 1
            if len(buf) >= 50000:
                _flush()
    _flush()
    if writer is not None:
        writer.close()

    odir = f"{a.out}/exec_outputs/domain=mlcb/language={a.language}"
    os.makedirs(odir, exist_ok=True)
    orows = [{"output_hash": oh, "output_kind": "stdio", "output": txt, "size": len(txt or "")}
             for oh, txt in out_rows.items()]
    out_parquet = f"{odir}/{a.cell}.parquet"
    pq.write_table(pa.Table.from_pylist(orows, schema=OUTPUT_SCHEMA), out_parquet, compression="zstd")
    print(f"  executions={n_exec} rows; exec_outputs={len(out_rows)} unique", flush=True)

    if a.upload_repo:
        _upload(a.upload_repo, [
            (exec_path, f"executions/domain=mlcb/language={a.language}/model={a.model}/temp={a.temp}/data.parquet",
             f"mlcb {a.language} {a.model} t{a.temp}: executions ({n_exec} rows)"),
            (out_parquet, f"exec_outputs/domain=mlcb/language={a.language}/{a.cell}.parquet",
             f"mlcb {a.language} {a.model} t{a.temp}: exec_outputs ({len(out_rows)} unique)"),
        ])
        if a.sentinel_dir:
            open(f"{a.sentinel_dir}/.captured_{a.cell}", "w").close()
    print("MLCB_CAPTURE_DONE", flush=True)


# ---------------------------------------------------------------------------- problem tables
def build_problems(a):
    insts = list(MultilingualLCBDataset.from_jsonl(a.snapshot, a.language))
    prows = []
    inputs_dim = {}
    for inst in insts:
        try:
            io = json.loads(inst.eval_sample["input_output"])
            ins, outs = list(io["inputs"]), list(io["outputs"])
        except Exception:
            ins, outs = [], []
        ih = [_sha(s) for s in ins]
        eh = [_sha(s) for s in outs]
        prows.append({
            "question_id": inst.question_id, "instance_id": inst.instance_id,
            "difficulty": getattr(inst, "difficulty", None), "platform": getattr(inst, "platform", None),
            "testtype": getattr(inst, "testtype", None),
            "contest_date": str(getattr(inst, "contest_date", "") or ""),
            "question_content": getattr(inst, "question_content", None),
            "starter_code": getattr(inst, "starter_code", None),
            "n_tests": len(outs), "input_hashes": ih, "expected_hashes": eh,
        })
        for s, h in zip(ins, ih):
            inputs_dim.setdefault(h, ("stdio", s))
        for s, h in zip(outs, eh):
            inputs_dim.setdefault(h, ("expected", s))

    pdir = f"{a.out}/problems/domain=mlcb/language={a.language}"
    os.makedirs(pdir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(prows, schema=PROB_SCHEMA), f"{pdir}/data.parquet", compression="zstd")
    idir = f"{a.out}/exec_inputs/domain=mlcb/language={a.language}"
    os.makedirs(idir, exist_ok=True)
    irows = [{"input_hash": h, "input_kind": kind, "input": txt, "size": len(txt or "")}
             for h, (kind, txt) in inputs_dim.items()]
    pq.write_table(pa.Table.from_pylist(irows, schema=INPUT_SCHEMA), f"{idir}/data.parquet", compression="zstd")
    print(f"problems: {len(insts)} rows; exec_inputs: {len(inputs_dim)} unique", flush=True)

    if a.upload_repo:
        _upload(a.upload_repo, [
            (f"{pdir}/data.parquet", f"problems/domain=mlcb/language={a.language}/data.parquet",
             f"mlcb {a.language}: problem definitions ({len(insts)})"),
            (f"{idir}/data.parquet", f"exec_inputs/domain=mlcb/language={a.language}/data.parquet",
             f"mlcb {a.language}: exec_inputs ({len(inputs_dim)} unique)"),
        ])
    print("BUILD_MLCB_PROBLEMS_DONE", flush=True)


def _upload(repo, files):
    from huggingface_hub import HfApi
    api = HfApi()
    for local, remote, msg in files:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=repo,
                        repo_type="dataset", commit_message=msg)
    print("  UPLOADED to", repo, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=["cell", "problems"])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--language", required=True)
    ap.add_argument("--out", default="/cluster/scratch/skiegeland/mlcb_exec_stage")
    ap.add_argument("--upload-repo", default="", help="HF dataset repo to upload to (empty = stage only)")
    # cell mode
    ap.add_argument("--cell", help="cell name, e.g. mlcb_qwen3-8b-nothink_t0.6")
    ap.add_argument("--completions", nargs="+", default=[], help="glob(s) of completion jsonl shards")
    ap.add_argument("--model", help="model tag incl arm")
    ap.add_argument("--temp")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-completion-seconds", type=float, default=1e9,
                    help="per-completion wall cap (bounds runaway loops); off by default, "
                         "the per-test timeout already bounds each test")
    ap.add_argument("--sentinel-dir", default="", help="touch .captured_<cell> here after upload")
    a = ap.parse_args()
    if a.mode == "problems":
        build_problems(a)
    else:
        for req in ("cell", "model", "temp"):
            if not getattr(a, req):
                ap.error(f"--{req} required in cell mode")
        if not a.completions:
            ap.error("--completions required in cell mode")
        capture_cell(a)


if __name__ == "__main__":
    main()
