"""Consistency of the multilingual-LCB domain with the default LCB evaluator and genlm-rollouts.

genlm-rollouts grades python through the default-LCB harness (``passed_all`` via
``testing_util.run_test``) and buckets failures into a fixed taxonomy with a matching pass@k
estimator. This module pins the interop contract and, importantly, the one place the two python
graders intentionally disagree:

  Our multilingual python path uses the Multi-LCB lenient comparator (the locked uniform-grading
  decision), which aliases True/true and False/false and allows a 1e-5 float tolerance. The
  default-LCB grader uses exact Decimal equality with no aliasing. So our python pass@1 is >= the
  default-LCB pass@1 that the rollouts leaderboard reports; multilingual-python numbers are not
  directly comparable to it. These tests make that gap visible and catch any future drift.

Where genlm-rollouts is importable (its repo sits beside genlm-eval) the tests assert directly
against the real taxonomy and estimator; otherwise those checks skip.
"""

import json
import math
import sys
from pathlib import Path

import pytest

from genlm.eval.domains.livecodebench.livecodebench import passed_all
from genlm.eval.domains.livecodebench_multilingual import (
    ROLLOUTS_CATEGORIES,
    LocalSubprocessExecutor,
    pass_at_k,
    rollouts_category,
    rollouts_error_code,
)
from genlm.eval.domains.livecodebench_multilingual.vendored import testing_plang as tp

_ROLLOUTS = Path.home() / "genlm-rollouts"


def _import_rollouts():
    """Import (CATEGORIES, pass_at_k) from the sibling genlm-rollouts repo, or (None, None)."""
    if not _ROLLOUTS.exists():
        return None, None
    if str(_ROLLOUTS) not in sys.path:
        sys.path.insert(0, str(_ROLLOUTS))
    try:
        from rollouts.analysis.lcb.taxonomy import CATEGORIES
        from rollouts.common.stats import pass_at_k as r_pass_at_k

        return CATEGORIES, r_pass_at_k
    except Exception:  # noqa: BLE001 - rollouts not installed here, so skip those checks
        return None, None


_RL_CATEGORIES, _RL_PASS_AT_K = _import_rollouts()

# ------------------------------ pass@k equivalence ------------------------------ #

# Boundary-covering cases for the estimator (no full grid: the closed form has few branches).
# Covers c=0, c=n, n-c<k early-return, k=1, k=n, and mixed interiors.
_PASS_AT_K_GRID = [
    (1, 0, 1),
    (1, 1, 1),
    (5, 0, 1),
    (5, 5, 1),
    (5, 5, 5),
    (5, 2, 1),
    (5, 2, 3),
    (5, 4, 5),
    (10, 3, 1),
    (10, 3, 10),
    (10, 1, 5),
    (8, 7, 8),
]


@pytest.mark.parametrize("n,c,k", _PASS_AT_K_GRID)
def test_pass_at_k_matches_combinatorial_reference(n, c, k):
    # Our product-form estimator must equal the closed form 1 - C(n-c,k)/C(n,k) that rollouts uses.
    expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected, abs=1e-12)


@pytest.mark.skipif(_RL_PASS_AT_K is None, reason="genlm-rollouts not importable")
@pytest.mark.parametrize("n,c,k", _PASS_AT_K_GRID)
def test_pass_at_k_matches_rollouts(n, c, k):
    # On the valid domain (1 <= k <= n) the two implementations agree exactly. They differ only on
    # edge inputs by design: ours raises on n<1 / k<=0 / c out of range, rollouts returns nan/1.0.
    assert pass_at_k(n, c, k) == pytest.approx(_RL_PASS_AT_K(n, c, k), abs=1e-12)


# ------------------------------ status to taxonomy interop ------------------------------ #

# Pinned mapping for every Status the executor can emit (plus the "ok" success string).
# UNK/Done/BuildDone are non-failures per Status.is_failure(), so they have no failure category.
_CATEGORY_CASES = {
    "ok": None,
    "UNK": None,
    "Done": None,
    "BuildDone": None,
    "EmptyCode": "no_code",
    "WrongAnswer": "wrong_answer",
    "TimeoutExpired": "timeout",
    "BuildFailed": "compile_error",
    "BuildTimeOut": "compile_error",
    "SyntaxError": "compile_error",
    "NPMFailed": "compile_error",
    "Exception": "runtime_error",
    "AbnormalTermination": "runtime_error",
    "OutOfMemory": "runtime_error",
    "ValueError": "runtime_error",
}


@pytest.mark.parametrize("status,expected", list(_CATEGORY_CASES.items()))
def test_rollouts_category_mapping(status, expected):
    assert rollouts_category(status) == expected
    if expected is not None:
        assert expected in ROLLOUTS_CATEGORIES


def test_every_executor_status_maps_to_a_known_category():
    # Every real Status enum name resolves (no KeyError) to None or a known category, so a
    # multilingual run can always be folded into the rollouts taxonomy.
    for member in tp.Status:
        cat = rollouts_category(member.name)
        assert cat is None or cat in ROLLOUTS_CATEGORIES


def test_every_executor_status_maps_to_a_known_error_code():
    # Same totality guard for the executions error code: every status lands in the rollouts set.
    for member in tp.Status:
        assert rollouts_error_code(member.name) in {1, -1, -2, -3, -4, -5}


def test_pass_statuses_match_is_failure():
    # The compat layer's "not a failure" set must agree with the executor's own Status.is_failure(),
    # so a status the executor counts as a pass is never mislabeled as a failure category (and vice
    # versa). "ok" is the success string the executor emits and has no enum member.
    non_failures = {m.name for m in tp.Status if not m.is_failure()}
    for member in tp.Status:
        is_pass = rollouts_category(member.name) is None
        assert is_pass == (member.name in non_failures), member.name


@pytest.mark.skipif(_RL_CATEGORIES is None, reason="genlm-rollouts not importable")
def test_categories_match_rollouts_repo():
    # Guards against drift: if either side renames/adds a bucket, this breaks.
    assert set(ROLLOUTS_CATEGORIES) == set(_RL_CATEGORIES)


_ERROR_CODE_CASES = {
    "ok": 1,
    "UNK": 1,
    "WrongAnswer": -2,
    "TimeoutExpired": -3,
    "Exception": -4,
    "AbnormalTermination": -4,
    "OutOfMemory": -4,
    "ValueError": -4,
    "BuildFailed": -5,
    "SyntaxError": -5,
    "BuildTimeOut": -5,
    "NPMFailed": -5,
    "EmptyCode": -1,
}


@pytest.mark.parametrize("status,code", list(_ERROR_CODE_CASES.items()))
def test_rollouts_error_code_mapping(status, code):
    assert rollouts_error_code(status) == code


def test_error_code_does_not_alias_execfail():
    # The per-test EXECFAIL value (-5) collapses timeout/runtime/compile; the error-code mapping
    # must split them by status so timeout != runtime != compile in the executions table.
    assert tp.TestScore.EXECFAIL.value == -5
    assert rollouts_error_code("TimeoutExpired") == -3
    assert rollouts_error_code("Exception") == -4
    assert rollouts_error_code("BuildFailed") == -5
    assert {rollouts_error_code(s) for s in _ERROR_CODE_CASES} <= {
        1,
        -1,
        -2,
        -3,
        -4,
        -5,
    }


def test_status_str_is_name_stable():
    # executor.py builds metadata["status"] from str(meta.error); it must equal the enum member
    # name on this Python (3.11 changed str(Enum)). TestScore.name backs per_test the same way.
    for member in tp.Status:
        assert str(member) == member.name
    for member in tp.TestScore:
        assert str(member) == member.name


# ------------------------------ python grader divergence ------------------------------ #


def _ours_solves(printed, expected):
    """Grade a python program that writes `printed` against `expected`, via our lenient path."""
    code = f"import sys\nsys.stdout.write({printed!r})\n"
    solved, _ = LocalSubprocessExecutor().run(code, ["x\n"], [expected], "python", 5.0)
    return solved


def _default_solves(printed, expected):
    """Grade the same program via the default-LCB harness genlm-rollouts uses."""
    code = f"import sys\nsys.stdout.write({printed!r})\n"
    es = {
        "input_output": json.dumps(
            {"inputs": ["x\n"], "outputs": [expected], "fn_name": None}
        )
    }
    return passed_all(es, code)


# (printed, expected): cases where lenient (ours) accepts but exact Decimal (default) rejects.
_DIVERGE = [
    ("true\n", "True\n"),
    ("false\n", "False\n"),
    ("1.000001\n", "1.000002\n"),  # within abs_tol=1e-5
]


@pytest.mark.parametrize("printed,expected", _DIVERGE)
def test_python_lenient_more_permissive_than_default(printed, expected):
    assert _ours_solves(printed, expected) is True
    assert _default_solves(printed, expected) is False


# Cases where both graders agree (anchors the divergence to exactly the three classes above).
_AGREE_PASS = [("6\n", "6\n"), ("6 \n", "6\n"), ("5\n", "5.0\n")]
_AGREE_FAIL = [("1.0001\n", "1.0002\n"), ("6\n7\n", "6\n")]


@pytest.mark.parametrize("printed,expected", _AGREE_PASS)
def test_python_graders_agree_on_pass(printed, expected):
    assert _ours_solves(printed, expected) is True
    assert _default_solves(printed, expected) is True


@pytest.mark.parametrize("printed,expected", _AGREE_FAIL)
def test_python_graders_agree_on_fail(printed, expected):
    assert _ours_solves(printed, expected) is False
    assert _default_solves(printed, expected) is False
