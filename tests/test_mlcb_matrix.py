"""Per-language verdict matrix for multilingual LiveCodeBench.

For every (language, variant) the executor must produce the contracted verdict: not just the
0/1 score, but the per-test list and the status class, so a grader that conflates a crash with
a wrong answer (or stops at the first pass) fails here. Snippets live in fixtures/mlcb_solutions
and were verified once against the real executor; this test re-confirms them. Languages whose
toolchain is absent skip (the default CI lane skips all compiled languages; the full lane with
the mlcb-tools env runs them).
"""

import json

import pytest
from fixtures.mlcb_solutions import SOLUTIONS, SUM_N_INPUTS, SUM_N_OUTPUTS

from genlm.eval.domains.livecodebench_multilingual import (
    LocalSubprocessExecutor,
    MultilingualLCBEvaluator,
    MultilingualLCBInstance,
    is_toolchain_available,
    resolve_language,
)

# Per-variant contract: (expected solved, allowed status strings, expected per_test list).
# per_test is a prefix that short-circuits at the first failing test.
CONTRACT = {
    "correct": (True, {"ok"}, ["PASSED", "PASSED", "PASSED"]),
    "wrong_output": (False, {"WrongAnswer"}, ["FAILED"]),
    "partial": (False, {"WrongAnswer"}, ["PASSED", "PASSED", "FAILED"]),
    "runtime_error": (False, {"Exception", "AbnormalTermination"}, ["EXECFAIL"]),
    "compile_error": (
        False,
        {"SyntaxError", "BuildFailed", "Exception", "AbnormalTermination"},
        ["EXECFAIL"],
    ),
}

_CASES = [(lang, variant) for lang, vs in SOLUTIONS.items() for variant in vs]


@pytest.mark.parametrize("language,variant", _CASES)
def test_verdict_matrix(language, variant):
    if not is_toolchain_available(language):
        pytest.skip(f"{language} toolchain not installed")
    expect_solved, allowed_status, expect_per_test = CONTRACT[variant]
    ex = LocalSubprocessExecutor()
    ex.prepare(language)  # idempotent; go/julia setup
    solved, meta = ex.run(
        SOLUTIONS[language][variant],
        SUM_N_INPUTS,
        SUM_N_OUTPUTS,
        language,
        timeout=20.0,
    )
    assert solved is expect_solved, f"{language}/{variant}: solved={solved} meta={meta}"
    assert meta["per_test"] == expect_per_test, f"{language}/{variant}: {meta}"
    assert meta["status"] in allowed_status, (
        f"{language}/{variant}: status={meta['status']}"
    )
    assert meta["n_tests"] == len(SUM_N_OUTPUTS)


@pytest.mark.parametrize("language", ["python", "c++"])
def test_evaluator_end_to_end_fenced(language):
    # The matrix drives the executor on raw code; this drives the full evaluator (extract_code
    # from a fenced generation, then grade), confirming the prompt-to-verdict path per language.
    if not is_toolchain_available(language):
        pytest.skip("toolchain absent")
    fence = resolve_language(language).md_fence
    inst = MultilingualLCBInstance(
        instance_id=f"sum@{language}",
        question_id="sum",
        language=language,
        question_content="sum of N integers",
        eval_sample={
            "input_output": json.dumps(
                {"inputs": SUM_N_INPUTS, "outputs": SUM_N_OUTPUTS, "fn_name": None}
            )
        },
    )
    ev = MultilingualLCBEvaluator(timeout_seconds=20.0)
    good = f"```{fence}\n{SOLUTIONS[language]['correct']}\n```"
    wrong = f"```{fence}\n{SOLUTIONS[language]['wrong_output']}\n```"
    assert ev.evaluate_sample(inst, good).score == 1.0
    assert ev.evaluate_sample(inst, wrong).score == 0.0
