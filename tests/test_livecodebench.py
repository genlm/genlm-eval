import base64
import importlib.util
import json
import pickle
import zlib
from pathlib import Path

import pytest

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
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE = FIXTURE_DIR / "lcb_sample.jsonl"


def _load_solutions():
    spec = importlib.util.spec_from_file_location("lcb_solutions", FIXTURE_DIR / "lcb_solutions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SOLUTIONS, mod.WRONG


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


def test_extract_last_fenced_block():
    out = "thinking...\n```python\nprint(1)\n```\nmore\n```python\nprint(2)\n```\n"
    assert extract_code(out).strip() == "print(2)"


def test_extract_no_fence_returns_empty():
    # Official lcb_runner extract_code returns "" when there are <2 fences.
    assert extract_code("  print(1)  ") == ""


def test_extract_handles_bare_fence():
    assert extract_code("```\nprint(3)\n```").strip() == "print(3)"


def test_extract_with_three_plus_fences_returns_last_block():
    # 3+ fences (prose ``` then a real block): returns the last block, per lcb_runner.
    out = "see ```inline``` then\n```python\nprint(42)\n```\n"
    assert extract_code(out).strip() == "print(42)"


# ------------------------------ harness ------------------------------ #

def test_vendored_run_test_importable():
    from genlm.eval.domains.livecodebench.util.testing_util import run_test
    assert callable(run_test)


STDIN_SAMPLE = {"input_output": json.dumps(
    {"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"], "fn_name": None})}
STDIN_GOOD = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n"
STDIN_BAD = "import sys\nn = int(sys.stdin.readline())\nprint(n + 1)\n"
FUNC_SAMPLE = {"input_output": json.dumps(
    {"inputs": ["3", "10"], "outputs": ["6", "20"], "fn_name": "double"})}
FUNC_GOOD = "class Solution:\n    def double(self, x):\n        return 2 * x\n"
FUNC_BAD = "class Solution:\n    def double(self, x):\n        return x + 1\n"


def test_stdin_correct_passes():
    assert passed_all(STDIN_SAMPLE, STDIN_GOOD, timeout=6.0) is True


def test_stdin_wrong_fails():
    assert passed_all(STDIN_SAMPLE, STDIN_BAD, timeout=6.0) is False


def test_malformed_eval_sample_fails_gracefully():
    # prompts-only / malformed snapshots must score fail, not crash the eval run
    for bad in ({}, {"input_output": "{}"}, {"input_output": "not json"}):
        assert passed_all(bad, STDIN_GOOD, timeout=6.0) is False
        assert check_correctness(bad, STDIN_GOOD, timeout=6.0)[0] == [-1]


def test_functional_correct_passes():
    assert passed_all(FUNC_SAMPLE, FUNC_GOOD, timeout=6.0) is True


def test_functional_wrong_fails():
    assert passed_all(FUNC_SAMPLE, FUNC_BAD, timeout=6.0) is False


def test_check_correctness_returns_per_test_list():
    results, _ = check_correctness(STDIN_SAMPLE, STDIN_GOOD, timeout=6.0)
    assert len(results) == 2 and all(r == 1 for r in results)


# --------------------------- evaluator / critic --------------------------- #

GOOD_GEN = "Here is my solution:\n```python\nimport sys\nn=int(sys.stdin.readline())\nprint(n*2)\n```\n"
BAD_GEN = "```python\nimport sys\nn=int(sys.stdin.readline())\nprint(n+1)\n```"


def _instance(eval_sample, qid="t"):
    return LiveCodeBenchInstance(instance_id=qid, question_id=qid,
                                 question_content="x", eval_sample=eval_sample)


def test_evaluator_scores_correct_generation_one():
    ev = LiveCodeBenchEvaluator(timeout_seconds=6.0)
    assert ev.evaluate_sample(_instance(STDIN_SAMPLE), GOOD_GEN).score == 1.0


def test_evaluator_scores_wrong_generation_zero():
    ev = LiveCodeBenchEvaluator(timeout_seconds=6.0)
    assert ev.evaluate_sample(_instance(STDIN_SAMPLE), BAD_GEN).score == 0.0


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
    inst = LiveCodeBenchInstance(instance_id="x", question_id="x",
                                 question_content="q", eval_sample={"input_output": "{}"})
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


# ------------------------------ dataset / split ------------------------------ #

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
    return [inst.question_id for inst in ds]


def test_split_is_disjoint_and_covers_all(tmp_path):
    snap = _write_snapshot(tmp_path)
    train = set(_ids(LiveCodeBenchDataset.from_jsonl(snap, split="train", test_frac=0.3, seed=7)))
    test = set(_ids(LiveCodeBenchDataset.from_jsonl(snap, split="test", test_frac=0.3, seed=7)))
    assert train.isdisjoint(test)
    assert train | test == {f"f{i}" for i in range(10)} | {f"s{i}" for i in range(10)}


def test_split_is_seed_stable(tmp_path):
    snap = _write_snapshot(tmp_path)
    a = _ids(LiveCodeBenchDataset.from_jsonl(snap, split="train", seed=7))
    b = _ids(LiveCodeBenchDataset.from_jsonl(snap, split="train", seed=7))
    assert a == b


def test_split_stratifies_testtype(tmp_path):
    snap = _write_snapshot(tmp_path)
    tts = [i.testtype for i in LiveCodeBenchDataset.from_jsonl(snap, split="test", test_frac=0.3, seed=7)]
    assert tts.count("functional") == 3 and tts.count("stdin") == 3


def test_difficulties_filter(tmp_path):
    snap = _write_snapshot(tmp_path)
    ds = LiveCodeBenchDataset.from_jsonl(snap, split="train", difficulties=["easy"])
    assert {i.difficulty for i in ds} == {"easy"}


def test_full_window_has_no_split(tmp_path):
    snap = _write_snapshot(tmp_path)
    assert len(LiveCodeBenchDataset.from_jsonl(snap)) == 20  # split=None -> all rows


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


def test_max_instances_caps(tmp_path):
    snap = _write_snapshot(tmp_path)
    assert len(LiveCodeBenchDataset.from_jsonl(snap, split="train", max_instances=5)) == 5


# ------------------------------ committed fixture ------------------------------ #

_ROWS = [json.loads(line) for line in SAMPLE.open() if line.strip()] if SAMPLE.exists() else []


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
def test_fixture_spans_both_testtypes_and_releases():
    assert {r["testtype"] for r in _ROWS} == {"stdin", "functional"}
    assert len({r["release"] for r in _ROWS}) >= 2


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
@pytest.mark.parametrize("row", _ROWS, ids=lambda r: r["question_id"])
def test_reference_solution_passes(row):
    sols, _ = _load_solutions()
    ev = LiveCodeBenchEvaluator(timeout_seconds=10.0)
    inst = _instance(row["eval_sample"], qid=row["question_id"])
    assert ev.evaluate_sample(inst, sols[row["question_id"]]).score == 1.0


@pytest.mark.skipif(not _ROWS, reason="fixture missing")
@pytest.mark.parametrize("row", _ROWS, ids=lambda r: r["question_id"])
def test_wrong_solution_fails(row):
    _, wrong = _load_solutions()
    ev = LiveCodeBenchEvaluator(timeout_seconds=10.0)
    inst = _instance(row["eval_sample"], qid=row["question_id"])
    assert ev.evaluate_sample(inst, wrong).score == 0.0
