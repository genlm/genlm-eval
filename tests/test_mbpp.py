"""MBPP domain: execution/classification, extraction, evaluator, dataset, and potentials.

The execution, extraction, evaluator, and dataset tests are pure-Python (no ML stack, no
network). The potential tests require ``genlm.control`` and are skipped if it is unavailable.
"""
import asyncio

import pytest

from genlm.eval.domains.mbpp import (
    MBPPDataset,
    MBPPEvaluator,
    build_prompt,
    extract_code,
    extract_code_prefix,
    run_mbpp,
)

TEXT = "Write a function to add two numbers."
TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]
CORRECT = "def add(a, b):\n    return a + b\n"
WRONG = "def add(a, b):\n    return a - b\n"          # loads + runs, wrong answer
RUNTIME_ERR = "def add(a, b):\n    return a + c\n"     # NameError when called
SYNTAX_ERR = "def add(a, b) return a + b\n"            # does not parse


# ------------------------------ extraction ------------------------------ #

def test_extract_code_fenced_and_plain():
    assert extract_code("```python\n" + CORRECT + "```") == CORRECT.strip()
    assert extract_code(CORRECT) == CORRECT.strip()
    # last fence wins; reasoning before </think> is dropped
    txt = "<think>```python\nx=1\n```</think>\nhere:\n```python\n" + CORRECT + "```"
    assert extract_code(txt) == CORRECT.strip()


def test_extract_code_prefix_handles_open_fence():
    partial = "Sure:\n```python\ndef add(a, b):\n    return a"
    assert extract_code_prefix(partial) == "def add(a, b):\n    return a"


# ------------------------------ run_mbpp ------------------------------ #

def test_run_mbpp_correct():
    r = run_mbpp(CORRECT, TESTS)
    assert r.loaded and r.all_passed and r.no_error
    assert r.n_passed == 2 and r.n_tests == 2 and r.n_runtime_errors == 0


def test_run_mbpp_wrong_answer_is_noerror_but_not_passing():
    r = run_mbpp(WRONG, TESTS)
    assert r.no_error is True          # it ran; a failed assert is not a runtime error
    assert r.all_passed is False
    assert r.n_passed == 0


def test_run_mbpp_runtime_error_is_not_noerror():
    r = run_mbpp(RUNTIME_ERR, TESTS)
    assert r.no_error is False
    assert r.n_runtime_errors == 2 and r.n_passed == 0


def test_run_mbpp_syntax_error():
    r = run_mbpp(SYNTAX_ERR, TESTS)
    assert r.syntax_error is True and r.no_error is False and r.all_passed is False


def test_run_mbpp_empty():
    r = run_mbpp("", TESTS)
    assert r.no_error is False and r.load_error == "EmptySolution"


def test_run_mbpp_uses_setup_code():
    code = "def area(r):\n    return math.pi * r * r\n"
    tests = ["assert abs(area(1) - 3.14159) < 1e-4"]
    assert run_mbpp(code, tests, test_setup_code="import math").all_passed
    # without the setup import, the call raises NameError -> not no-error
    assert run_mbpp(code, tests).no_error is False


def test_run_mbpp_timeout():
    code = "def add(a, b):\n    while True:\n        pass\n"
    r = run_mbpp(code, TESTS, timeout_seconds=1.0)
    assert r.timeout is True


# ------------------------------ evaluator ------------------------------ #

def _inst(**kw):
    row = {"task_id": 1, "text": TEXT, "test_list": TESTS, "code": CORRECT, **kw}
    return next(iter(MBPPDataset([row])))


def test_evaluator_scores_correct_and_wrong():
    ev = MBPPEvaluator()
    assert ev.evaluate_sample(_inst(), "```python\n" + CORRECT + "```").score == 1.0
    res = ev.evaluate_sample(_inst(), "```python\n" + WRONG + "```")
    assert res.score == 0.0 and res.metadata["n_passed"] == 0 and res.metadata["task_id"] == 1


# ------------------------------ dataset ------------------------------ #

def test_dataset_full_and_sanitized_rows():
    inst = _inst(test_setup_code="import math")
    assert inst.instance_id == 1 and inst.test_list == TESTS
    assert inst.test_setup_code == "import math"
    assert "add(2, 3) == 5" in inst.prompt and TEXT in inst.prompt
    # sanitized layout: prompt/test_imports instead of text/test_setup_code
    row = {"task_id": 7, "prompt": TEXT, "test_list": TESTS, "test_imports": ["import math"]}
    sinst = next(iter(MBPPDataset([row], config="sanitized")))
    assert sinst.text == TEXT and sinst.test_setup_code == "import math" and sinst.config == "sanitized"


def test_build_prompt_includes_tests():
    p = build_prompt(TEXT, TESTS)
    assert TEXT in p and all(t in p for t in TESTS)


# ------------------------------ potentials (need genlm.control) ------------------------------ #

def _complete(pot, generation):
    return asyncio.get_event_loop().run_until_complete(pot.complete(generation))


def test_noerror_potential():
    pytest.importorskip("genlm.control")
    from genlm.eval.domains.mbpp import MBPPRuntimeNoErrorPotential

    pot = MBPPRuntimeNoErrorPotential(test_list=TESTS)
    assert _complete(pot, "```python\n" + CORRECT + "```") == 0.0
    assert _complete(pot, "```python\n" + WRONG + "```") == 0.0            # wrong but ran
    assert _complete(pot, "```python\n" + RUNTIME_ERR + "```") == float("-inf")
    assert _complete(pot, "```python\n" + SYNTAX_ERR + "```") == float("-inf")


def test_test_passing_potential_soft_and_hard():
    pytest.importorskip("genlm.control")
    from genlm.eval.domains.mbpp import MBPPTestPassingPotential

    soft = MBPPTestPassingPotential(test_list=TESTS, penalty_per_failed=2.0)
    assert _complete(soft, "```python\n" + CORRECT + "```") == 0.0
    s = _complete(soft, "```python\n" + WRONG + "```")
    assert s == pytest.approx(-4.0) and s > float("-inf")                  # 2 failed * 2.0

    hard = MBPPTestPassingPotential(test_list=TESTS, hard=True)
    assert _complete(hard, "```python\n" + CORRECT + "```") == 0.0
    assert _complete(hard, "```python\n" + WRONG + "```") == float("-inf")
