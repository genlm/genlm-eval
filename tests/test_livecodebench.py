import base64
import json
import pickle
import zlib
from pathlib import Path

import pytest
from fixtures.lcb_solutions import SOLUTIONS, WRONG

from genlm.eval.domains.livecodebench import (
    LiveCodeBenchDataset,
    LiveCodeBenchEvaluator,
    LiveCodeBenchInstance,
    format_lcb_prompt,
    extract_code,
    check_correctness,
    passed_all,
    build_row,
    derive_testtype,
    iter_release_rows,
)
from genlm.eval.domains.livecodebench.fetch import _release_num

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURE_DIR / "lcb_sample.jsonl"


# ------------------------------ prompts ------------------------------ #

def test_stdin_prompt_has_stdin_instruction():
    p = format_lcb_prompt({"question_content": "Print n*2.", "starter_code": ""})
    assert "Print n*2." in p and "stdin" in p.lower() and "```python" in p
    # official lcb_runner structure
    assert "### Question:" in p and "### Format:" in p and "### Answer:" in p


def test_functional_prompt_includes_starter_code():
    p = format_lcb_prompt({"question_content": "Implement double.",
                           "starter_code": "class Solution:\n    def double(self, x):"})
    assert "class Solution" in p and "starter code" in p.lower()


def test_codeqwen_prompt_has_blank_line_after_user_marker():
    # Official lcb_runner joins SYSTEM_MESSAGE_CODEQWEN + "\n\n" + body, i.e. a blank
    # line after "<|im_start|>user". A single "\n" here is a real prompt bug for Qwen.
    p = format_lcb_prompt({"question_content": "Print n*2.", "starter_code": ""}, style="codeqwen")
    assert "<|im_start|>user\n\nYou will be given" in p
    assert p.rstrip().endswith("<|im_start|>assistant")


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="style"):
        format_lcb_prompt({"question_content": "x", "starter_code": ""}, style="qwen")


# extract_code mirrors lcb_runner: code between the last two fences, "" if <2 fences.
@pytest.mark.parametrize("out, expected", [
    ("thinking...\n```python\nprint(1)\n```\nmore\n```python\nprint(2)\n```\n", "print(2)"),  # last of 2 blocks
    ("  print(1)  ", ""),                                                 # <2 fences -> ""
    ("```\nprint(3)\n```", "print(3)"),                                   # bare fence (no language)
    ("see ```inline``` then\n```python\nprint(42)\n```\n", "print(42)"),  # 3+ fences -> last block
])
def test_extract_code(out, expected):
    assert extract_code(out).strip() == expected


# ------------------------------ harness ------------------------------ #

STDIN_SAMPLE = {"input_output": json.dumps(
    {"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"], "fn_name": None})}
STDIN_GOOD = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n"
STDIN_BAD = "import sys\nn = int(sys.stdin.readline())\nprint(n + 1)\n"
FUNC_SAMPLE = {"input_output": json.dumps(
    {"inputs": ["3", "10"], "outputs": ["6", "20"], "fn_name": "double"})}
FUNC_GOOD = "class Solution:\n    def double(self, x):\n        return 2 * x\n"
FUNC_BAD = "class Solution:\n    def double(self, x):\n        return x + 1\n"


@pytest.mark.parametrize("sample, sol, expected", [
    (STDIN_SAMPLE, STDIN_GOOD, True),
    (STDIN_SAMPLE, STDIN_BAD, False),
    (FUNC_SAMPLE, FUNC_GOOD, True),
    (FUNC_SAMPLE, FUNC_BAD, False),
])
def test_passed_all(sample, sol, expected):
    assert passed_all(sample, sol, timeout=6.0) is expected


def test_malformed_eval_sample_fails_gracefully():
    # prompts-only / malformed snapshots must score fail, not crash the eval run
    for bad in ({}, {"input_output": "{}"}, {"input_output": "not json"}):
        assert passed_all(bad, STDIN_GOOD, timeout=6.0) is False
        assert check_correctness(bad, STDIN_GOOD, timeout=6.0)[0] == [-1]


def test_check_correctness_returns_per_test_list():
    results, _ = check_correctness(STDIN_SAMPLE, STDIN_GOOD, timeout=6.0)
    assert len(results) == 2 and all(r == 1 for r in results)


SLEEPY_SAMPLE = {"input_output": json.dumps(
    {"inputs": ["3\n"], "outputs": ["6\n"], "fn_name": None})}
SLEEPY_GOOD = ("import sys, time\ntime.sleep(1.2)\n"
               "n = int(sys.stdin.readline())\nprint(n * 2)\n")


def test_fractional_timeout_rounds_up_not_down():
    # timeout=1.5 must give the child at least 2s (ceil), not 1s (int truncation):
    # a correct solution needing 1.2s would otherwise be graded TLE.
    assert passed_all(SLEEPY_SAMPLE, SLEEPY_GOOD, timeout=1.5) is True


# --------------------------- evaluator / critic --------------------------- #

GOOD_GEN = "Here is my solution:\n```python\nimport sys\nn=int(sys.stdin.readline())\nprint(n*2)\n```\n"
BAD_GEN = "```python\nimport sys\nn=int(sys.stdin.readline())\nprint(n+1)\n```"


def _instance(eval_sample, qid="t"):
    return LiveCodeBenchInstance(instance_id=qid, question_content="x", eval_sample=eval_sample)


@pytest.mark.parametrize("gen, expected", [(GOOD_GEN, 1.0), (BAD_GEN, 0.0)])
def test_evaluator_scores(gen, expected):
    ev = LiveCodeBenchEvaluator(timeout_seconds=6.0)
    assert ev.evaluate_sample(_instance(STDIN_SAMPLE), gen).score == expected


def test_evaluator_memoizes_identical_generations(monkeypatch):
    import genlm.eval.domains.livecodebench.livecodebench as lcb_mod
    calls = []
    real = lcb_mod.passed_all
    monkeypatch.setattr(lcb_mod, "passed_all",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    ev = LiveCodeBenchEvaluator(timeout_seconds=6.0)
    inst = _instance(STDIN_SAMPLE)
    for _ in range(3):  # identical particles must hit the harness only once
        assert ev.evaluate_sample(inst, GOOD_GEN).score == 1.0
    assert len(calls) == 1


class _FakeTok:
    """Records the add_special_tokens flag; apply_chat_template emits a BOS literal."""
    def __init__(self):
        self.last_add_special = None
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        return "<BOS>" + messages[-1]["content"]
    def encode(self, text, add_special_tokens=True):
        self.last_add_special = add_special_tokens
        return [1, 2, 3]


def test_prompt_formatter_avoids_double_bos_on_chat_path():
    from genlm.eval.domains.livecodebench import default_prompt_formatter as fmt
    inst = LiveCodeBenchInstance(instance_id="x", question_content="q",
                                 eval_sample={"input_output": "{}"})
    tok = _FakeTok()
    fmt(tok, inst, use_chat_format=True)
    assert tok.last_add_special is False    # chat template already added BOS
    fmt(tok, inst, use_chat_format=False)
    assert tok.last_add_special is True     # base path must add BOS


# ------------------------------ fetch decode ------------------------------ #

def _functional_raw():
    return {"question_content": "Return 2x.", "platform": "leetcode",
            "question_id": "q-func-1", "contest_date": "2023-05-01T00:00:00",
            "difficulty": "easy", "starter_code": "class Solution:\n    def double(self, x):",
            "public_test_cases": json.dumps([{"input": "3", "output": "6", "testtype": "functional"}]),
            "private_test_cases": json.dumps([{"input": "10", "output": "20", "testtype": "functional"}]),
            "metadata": json.dumps({"func_name": "double"})}


def _stdin_raw_compressed():
    private = [{"input": "10\n", "output": "20\n", "testtype": "stdin"}]
    blob = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(private)))).decode("utf-8")
    return {"question_content": "Print 2n.", "platform": "codeforces",
            "question_id": "q-stdin-1", "contest_date": "2024-09-01T00:00:00",
            "difficulty": "medium", "starter_code": "",
            "public_test_cases": json.dumps([{"input": "3\n", "output": "6\n", "testtype": "stdin"}]),
            "private_test_cases": blob, "metadata": json.dumps({})}


def test_build_row_functional_plain_json():
    row = build_row(_functional_raw(), release="release_v1")
    io = json.loads(row["eval_sample"]["input_output"])
    assert io["fn_name"] == "double" and io["inputs"] == ["3", "10"] and io["outputs"] == ["6", "20"]
    assert row["testtype"] == "functional" and row["difficulty"] == "easy" and row["release"] == "release_v1"


def test_build_row_stdin_compressed():
    row = build_row(_stdin_raw_compressed(), release="release_v6")
    io = json.loads(row["eval_sample"]["input_output"])
    assert io["fn_name"] is None and io["inputs"] == ["3\n", "10\n"] and row["testtype"] == "stdin"


def test_max_tests_truncates():
    row = build_row(_stdin_raw_compressed(), release="release_v6", max_tests=1)
    io = json.loads(row["eval_sample"]["input_output"])
    assert len(io["inputs"]) == 1 and len(io["outputs"]) == 1


def test_derive_testtype_uses_func_name():
    assert derive_testtype({"func_name": "f"}, []) == "functional"
    assert derive_testtype({}, [{"testtype": "stdin"}]) == "stdin"


def test_release_num_strict():
    assert _release_num("release_v1") == 1
    assert _release_num("release_v12") == 12
    for bad in ("release_v-1", "release_v0", "release_v1_2", "v6", "release_v", "release_v 2"):
        with pytest.raises(ValueError):
            _release_num(bad)


def test_iter_release_rows_raw_filter_skips_rows(tmp_path, monkeypatch):
    raw_file = tmp_path / "test.jsonl"
    raw_file.write_text(json.dumps(_functional_raw()) + "\n"           # 2023 row
                        + json.dumps(_stdin_raw_compressed()) + "\n")  # 2024 row
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(raw_file))
    rows = list(iter_release_rows("release_v1",
                                  raw_filter=lambda r: r["contest_date"] >= "2024"))
    assert [r["question_id"] for r in rows] == ["q-stdin-1"]


def _raw_min(qid, date="2024-01-01T00:00:00"):
    return {"question_id": qid, "question_content": f"Q {qid}.", "platform": "codeforces",
            "contest_date": date, "difficulty": "easy", "starter_code": "",
            "public_test_cases": json.dumps([{"input": "1\n", "output": "1\n", "testtype": "stdin"}]),
            "private_test_cases": json.dumps([]), "metadata": json.dumps({})}


def test_iter_release_rows_cumulative_dedupes_across_windows(tmp_path, monkeypatch):
    # qDUP appears in BOTH windows; cumulative load must yield it once, tagged to v1 (first seen).
    (tmp_path / "test.jsonl").write_text(json.dumps(_raw_min("qA")) + "\n" + json.dumps(_raw_min("qDUP")) + "\n")
    (tmp_path / "test2.jsonl").write_text(json.dumps(_raw_min("qDUP")) + "\n" + json.dumps(_raw_min("qB")) + "\n")
    files = {"test.jsonl": str(tmp_path / "test.jsonl"), "test2.jsonl": str(tmp_path / "test2.jsonl")}
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: files[kw["filename"]])

    rows = list(iter_release_rows("release_v2", cumulative=True))
    assert [r["question_id"] for r in rows] == ["qA", "qDUP", "qB"]  # each once, v1 before v2
    rel = {r["question_id"]: r["release"] for r in rows}
    assert rel["qDUP"] == "release_v1" and rel["qB"] == "release_v2"  # first-seen window wins

    # cumulative=False loads only the release_v2 window
    only = [r["question_id"] for r in iter_release_rows("release_v2", cumulative=False)]
    assert only == ["qDUP", "qB"]


# ------------------------------ dataset / holdout ------------------------------ #

def _write_snapshot(tmp_path: Path) -> Path:
    rows = []
    for i in range(10):
        rows.append({"question_id": f"f{i}", "question_content": f"func {i}",
                     "starter_code": "class Solution:\n    def f(self):",
                     "difficulty": ["easy", "medium", "hard"][i % 3], "platform": "leetcode",
                     "contest_date": "2023-05-01", "testtype": "functional", "release": "release_v1",
                     "eval_sample": {"input_output": json.dumps({"inputs": ["1"], "outputs": ["1"], "fn_name": "f"})},
                     "metadata": {"func_name": "f"}})
    for i in range(10):
        rows.append({"question_id": f"s{i}", "question_content": f"stdin {i}",
                     "starter_code": "", "difficulty": ["easy", "medium", "hard"][i % 3], "platform": "codeforces",
                     "contest_date": "2024-09-01", "testtype": "stdin", "release": "release_v6",
                     "eval_sample": {"input_output": json.dumps({"inputs": ["1\n"], "outputs": ["1\n"], "fn_name": None})},
                     "metadata": {}})
    p = tmp_path / "snap.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _ids(ds):
    return [inst.instance_id for inst in ds]


def test_holdout_is_disjoint_and_covers_all(tmp_path):
    snap = _write_snapshot(tmp_path)
    train = set(_ids(LiveCodeBenchDataset.from_jsonl(snap, holdout="train", test_frac=0.3, seed=7)))
    test = set(_ids(LiveCodeBenchDataset.from_jsonl(snap, holdout="test", test_frac=0.3, seed=7)))
    assert train.isdisjoint(test)
    assert train | test == {f"f{i}" for i in range(10)} | {f"s{i}" for i in range(10)}


def test_holdout_is_seed_stable(tmp_path):
    snap = _write_snapshot(tmp_path)
    a = _ids(LiveCodeBenchDataset.from_jsonl(snap, holdout="train", seed=7))
    b = _ids(LiveCodeBenchDataset.from_jsonl(snap, holdout="train", seed=7))
    assert a == b


def test_holdout_stratifies_testtype(tmp_path):
    snap = _write_snapshot(tmp_path)
    tts = [i.testtype for i in LiveCodeBenchDataset.from_jsonl(snap, holdout="test", test_frac=0.3, seed=7)]
    assert tts.count("functional") == 3 and tts.count("stdin") == 3


def test_invalid_holdout_raises_even_on_empty_data(tmp_path):
    snap = _write_snapshot(tmp_path)
    with pytest.raises(ValueError, match="holdout"):
        LiveCodeBenchDataset.from_jsonl(snap, holdout="vali")
    with pytest.raises(ValueError, match="holdout"):  # must not depend on rows surviving filters
        LiveCodeBenchDataset.from_jsonl(snap, holdout="vali", start_date="2999-01-01")


def test_difficulties_filter(tmp_path):
    snap = _write_snapshot(tmp_path)
    ds = LiveCodeBenchDataset.from_jsonl(snap, holdout="train", difficulties=["easy"])
    assert {i.difficulty for i in ds} == {"easy"}


def test_full_window_has_no_holdout(tmp_path):
    snap = _write_snapshot(tmp_path)
    assert len(LiveCodeBenchDataset.from_jsonl(snap)) == 20  # holdout=None -> all rows


def test_date_window_filter(tmp_path):
    snap = _write_snapshot(tmp_path)
    ds = LiveCodeBenchDataset.from_jsonl(snap, start_date="2024-01-01")
    assert {i.testtype for i in ds} == {"stdin"}  # only the 2024 stdin rows survive


def test_end_date_excludes_timed_contest_on_boundary_day(tmp_path):
    # datetime compare (matches official): a contest at 19:30 on the end_date day is
    # EXCLUDED (end_date parses to midnight); the next day includes it.
    row = {"question_id": "a", "question_content": "x", "starter_code": "",
           "difficulty": "easy", "platform": "p", "contest_date": "2024-08-17T19:30:00",
           "testtype": "stdin", "release": "release_v4",
           "eval_sample": {"input_output": "{}"}, "metadata": {}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(row) + "\n")
    assert len(list(LiveCodeBenchDataset.from_jsonl(p, end_date="2024-08-17"))) == 0
    assert len(list(LiveCodeBenchDataset.from_jsonl(p, end_date="2024-08-18"))) == 1


def test_timezone_aware_contest_date_does_not_crash(tmp_path):
    row = {"question_id": "tz", "question_content": "x", "starter_code": "",
           "difficulty": "easy", "platform": "p", "contest_date": "2024-08-17T19:30:00+00:00",
           "testtype": "stdin", "eval_sample": {"input_output": "{}"}}
    p = tmp_path / "s.jsonl"
    p.write_text(json.dumps(row) + "\n")
    assert len(list(LiveCodeBenchDataset.from_jsonl(p, end_date="2024-08-18"))) == 1
    assert len(list(LiveCodeBenchDataset.from_jsonl(p, end_date="2024-08-17"))) == 0


def test_max_instances_caps(tmp_path):
    snap = _write_snapshot(tmp_path)
    assert len(LiveCodeBenchDataset.from_jsonl(snap, holdout="train", max_instances=5)) == 5


def test_shuffle_is_seeded_permutation(tmp_path):
    snap = _write_snapshot(tmp_path)
    plain = _ids(LiveCodeBenchDataset.from_jsonl(snap))
    shuffled = _ids(LiveCodeBenchDataset.from_jsonl(snap, shuffle=True, seed=1))
    assert sorted(plain) == sorted(shuffled) and plain != shuffled
    assert shuffled == _ids(LiveCodeBenchDataset.from_jsonl(snap, shuffle=True, seed=1))


def test_to_jsonl_roundtrip(tmp_path):
    snap = _write_snapshot(tmp_path)
    ds = LiveCodeBenchDataset.from_jsonl(snap, start_date="2024-01-01")
    out = tmp_path / "out.jsonl"
    ds.to_jsonl(out)
    again = LiveCodeBenchDataset.from_jsonl(out)
    assert _ids(again) == _ids(ds)  # snapshot inherits the window; reload is identity


# ------------------------------ committed fixture ------------------------------ #

_ROWS = [json.loads(line) for line in SAMPLE.open() if line.strip()] if SAMPLE.exists() else []


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
def test_fixture_spans_both_testtypes_and_releases():
    assert {r["testtype"] for r in _ROWS} == {"stdin", "functional"}
    assert len({r["release"] for r in _ROWS}) >= 2


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
@pytest.mark.parametrize("row", _ROWS, ids=lambda r: r["question_id"])
def test_reference_solution_passes(row):
    ev = LiveCodeBenchEvaluator(timeout_seconds=10.0)
    inst = _instance(row["eval_sample"], qid=row["question_id"])
    assert ev.evaluate_sample(inst, SOLUTIONS[row["question_id"]]).score == 1.0


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
@pytest.mark.parametrize("row", _ROWS, ids=lambda r: r["question_id"])
def test_wrong_solution_fails(row):
    ev = LiveCodeBenchEvaluator(timeout_seconds=10.0)
    inst = _instance(row["eval_sample"], qid=row["question_id"])
    assert ev.evaluate_sample(inst, WRONG).score == 0.0
