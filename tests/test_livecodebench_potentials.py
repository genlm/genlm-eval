import json

import pytest

from genlm.eval.domains.livecodebench import (
    LCBPublicTestPotential,
    LCBRuntimeNoErrorPotential,
    LiveCodeBenchInstance,
    PublicTestFeedback,
    decode_context,
    extract_code_prefix,
    format_repair_prompt,
    repair_question_content,
)
from genlm.eval.domains.livecodebench import runtime_no_error_potential as rne

# fork()-based code execution under a torch-loaded (multi-threaded) parent emits
# a benign DeprecationWarning, exactly as the existing LCB harness does.
pytestmark = pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)

# --------------------------- samples --------------------------- #

PUB_FUNC = {"input_output": json.dumps(
    {"inputs": ["3", "10"], "outputs": ["6", "20"], "fn_name": "double"})}
FUNC_GOOD = "class Solution:\n    def double(self, x):\n        return 2 * x\n"
FUNC_RAISE = "class Solution:\n    def double(self, x):\n        return x[0]\n"  # int not subscriptable
FUNC_WRONG = "class Solution:\n    def double(self, x):\n        return x + 1\n"

PUB_STDIN = {"input_output": json.dumps(
    {"inputs": ["3\n", "10\n"], "outputs": ["6\n", "20\n"], "fn_name": None})}
STDIN_GOOD = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n"
STDIN_RAISE = "import sys\nn = int(sys.stdin.readline())\nprint(undefined_name)\n"
STDIN_WRONG = "import sys\nn = int(sys.stdin.readline())\nprint(n + 1)\n"


def _rt(public, **kw):
    kw.setdefault("extraction_style", "genericbase")
    kw.setdefault("timeout_seconds", 6.0)
    return LCBRuntimeNoErrorPotential(public_eval_sample=public, **kw)


# ===================== LCBRuntimeNoErrorPotential ===================== #

@pytest.mark.asyncio
@pytest.mark.parametrize("sample, code, expected", [
    (PUB_FUNC, FUNC_GOOD, 0.0),
    (PUB_FUNC, FUNC_RAISE, float("-inf")),
    (PUB_FUNC, FUNC_WRONG, 0.0),       # wrong answer tolerated: it still runs
    (PUB_STDIN, STDIN_GOOD, 0.0),
    (PUB_STDIN, STDIN_RAISE, float("-inf")),
    (PUB_STDIN, STDIN_WRONG, 0.0),
])
async def test_runtime_complete(sample, code, expected):
    assert await _rt(sample).complete([code.encode()]) == expected


@pytest.mark.asyncio
async def test_runtime_functional_prefix_syntax_only_no_forward_ref_kill():
    # Functional entrypoint calling a helper defined after it. complete() passes
    # and no prefix is killed, since functional prefixes only check syntax.
    pub = {"input_output": json.dumps({"inputs": ["6"], "outputs": ["x"], "fn_name": "f"})}
    code = ("class Solution:\n    def f(self, n):\n        return helper(n)\n"
            "\ndef helper(n):\n    return n * 2\n")
    pot = LCBRuntimeNoErrorPotential(public_eval_sample=pub, extraction_style="genericbase")
    assert await pot.complete([code.encode()]) == 0.0
    for i, ch in enumerate(code):
        if ch == "\n":
            assert await pot.prefix([code[: i + 1].encode()]) != float("-inf")


@pytest.mark.asyncio
async def test_runtime_complete_checks_all_public_inputs():
    # complete() must run every public input: a crash on a later input (not the
    # first) is still a runtime error, even though prefix only checks input 0.
    pub = {"input_output": json.dumps(
        {"inputs": ["3", "0"], "outputs": ["6", "0"], "fn_name": "f"})}
    crash_on_second = "class Solution:\n    def f(self, x):\n        return 10 // x\n"  # x=0 -> ZeroDivisionError
    pot = LCBRuntimeNoErrorPotential(public_eval_sample=pub, extraction_style="genericbase")
    assert await pot.complete([crash_on_second.encode()]) == float("-inf")


@pytest.mark.asyncio
async def test_runtime_complete_empty_is_fatal():
    pot = _rt(PUB_FUNC)
    assert await pot.complete([b""]) == float("-inf")
    assert await pot.complete([b"   \n  \n"]) == float("-inf")


@pytest.mark.asyncio
async def test_runtime_prefix_newline_guardrail():
    pot = _rt(PUB_STDIN)
    # No trailing newline, so nothing is judged (default line sampler).
    assert await pot.prefix([b"import nonexistent_module_xyz"]) == 0.0


@pytest.mark.asyncio
async def test_runtime_prefix_syntax_gating():
    pot = _rt(PUB_FUNC)
    # Incomplete (open def/class): a continuation can still close it.
    assert await pot.prefix([b"class Solution:\n    def double(self, x):\n"]) == 0.0
    assert pot.last_was_syntax_error is False
    # Unfixable syntax error: killed at the prefix already.
    assert await pot.prefix([b"def (:\n"]) == float("-inf")
    assert pot.last_was_syntax_error is True


@pytest.mark.asyncio
async def test_runtime_prefix_stdin_leading_statements():
    pot = _rt(PUB_STDIN)
    assert await pot.prefix([b"import sys\n"]) == 0.0                  # import-only ok
    assert await pot.prefix([b"import nonexistent_module_xyz\n"]) == float("-inf")
    assert await pot.prefix([b"x = undefined_name_xyz\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_runtime_prefix_functional_body_defers_until_complete():
    pot = _rt(PUB_FUNC)
    # The whole class is the trailing compound, so prefix drops it (defer); the
    # raising body is only caught at complete().
    assert await pot.prefix([FUNC_RAISE.encode()]) == 0.0
    assert await pot.complete([FUNC_RAISE.encode()]) == float("-inf")


@pytest.mark.asyncio
async def test_runtime_syntax_only_mode_without_public_inputs():
    pot = _rt({})  # no public sample, so syntax-only
    assert await pot.complete([FUNC_GOOD.encode()]) == 0.0
    assert await pot.complete([b"def (:\n"]) == float("-inf")
    assert await pot.prefix([b"def (:\n"]) == float("-inf")
    assert await pot.prefix([FUNC_RAISE.encode()]) == 0.0  # parses; not executed


@pytest.mark.asyncio
@pytest.mark.parametrize("sample, code", [
    (PUB_FUNC, FUNC_GOOD), (PUB_FUNC, FUNC_WRONG),
    (PUB_STDIN, STDIN_GOOD), (PUB_STDIN, STDIN_WRONG),
])
async def test_runtime_soundness_no_prefix_of_passing_code_killed(sample, code):
    # If complete() accepts the generation (0.0), no line-boundary prefix may be
    # killed (-inf). Holds for genericbase (code == raw output).
    pot = _rt(sample)
    assert await pot.complete([code.encode()]) == 0.0
    for i, ch in enumerate(code):
        if ch == "\n":
            assert await pot.prefix([code[: i + 1].encode()]) != float("-inf")


@pytest.mark.asyncio
async def test_runtime_timeout_defers_at_prefix_kills_at_complete():
    infloop = "while True:\n    pass\nx = 1\n"  # while is not the trailing stmt -> runs
    pot = _rt(PUB_STDIN, timeout_seconds=1.0)
    assert await pot.prefix([infloop.encode()]) == 0.0          # slow prefix -> defer
    assert await pot.complete([infloop.encode()]) == float("-inf")


@pytest.mark.asyncio
async def test_runtime_caches_and_skips_hung_prefix_extensions(monkeypatch):
    calls = []
    real = rne.run_noerror_check
    monkeypatch.setattr(rne, "run_noerror_check",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    pot = _rt(PUB_STDIN, timeout_seconds=1.0)
    infloop = "while True:\n    pass\nx = 1\n"
    assert await pot.prefix([infloop.encode()]) == 0.0
    n = len(calls)
    # Extending a hung prefix re-runs the same leading statements: defer without
    # paying the timeout again.
    assert await pot.prefix([(infloop + "y = 2\n").encode()]) == 0.0
    assert len(calls) == n


@pytest.mark.asyncio
async def test_runtime_verdict_cache_replays():
    pot = _rt(PUB_FUNC)
    assert await pot.complete([FUNC_RAISE.encode()]) == float("-inf")
    assert pot.cache_misses == 1 and pot.cache_hits == 0
    assert await pot.complete([FUNC_RAISE.encode()]) == float("-inf")
    assert pot.cache_hits == 1


@pytest.mark.asyncio
async def test_runtime_fenced_extraction_default_style():
    # Default "generic" style: code lives between ``` fences.
    pot = LCBRuntimeNoErrorPotential(public_eval_sample=PUB_FUNC, timeout_seconds=6.0)
    gen = f"Here is my solution:\n```python\n{FUNC_GOOD}```\n"
    assert await pot.complete([gen.encode()]) == 0.0
    bad = f"```python\n{FUNC_RAISE}```\n"
    assert await pot.complete([bad.encode()]) == float("-inf")


def test_runtime_coerce_adopts_vocab():
    from types import SimpleNamespace
    pot = _rt(PUB_FUNC)
    pot2 = pot.coerce(SimpleNamespace(vocab=[b"a", b"b"]))
    assert list(pot2.vocab) == [b"a", b"b"]
    assert pot2.public_eval_sample == PUB_FUNC


# ======================= LCBPublicTestPotential ======================= #

# Passes input 0 (x=3 -> 6) but fails input 1 (x=10 -> 11).
FUNC_HALF = "class Solution:\n    def double(self, x):\n        return 6 if x == 3 else x + 1\n"


def _pt(public=PUB_FUNC, **kw):
    kw.setdefault("extraction_style", "genericbase")
    kw.setdefault("timeout_seconds", 6.0)
    return LCBPublicTestPotential(public_eval_sample=public, **kw)


@pytest.mark.asyncio
async def test_public_prefix_never_kills():
    pot = _pt()
    for code in (FUNC_RAISE, "def (:\n", "", FUNC_WRONG):
        assert await pot.prefix([code.encode()]) == 0.0


@pytest.mark.asyncio
async def test_public_complete_all_pass_is_zero():
    assert await _pt().complete([FUNC_GOOD.encode()]) == 0.0


@pytest.mark.asyncio
async def test_public_complete_soft_penalty_is_finite():
    pot = _pt(penalty_per_failed=2.0, min_score=-10.0)
    score = await pot.complete([FUNC_WRONG.encode()])  # both public tests fail
    assert score == -4.0  # 2 failed * -2.0, above the floor
    half = await _pt().complete([FUNC_HALF.encode()])  # one of two fails
    assert half == -2.0


@pytest.mark.asyncio
async def test_public_complete_respects_floor():
    pot = _pt(penalty_per_failed=100.0, min_score=-5.0)
    assert await pot.complete([FUNC_WRONG.encode()]) == -5.0  # clamped, never -inf


@pytest.mark.asyncio
async def test_public_complete_no_public_sample_is_zero():
    assert await _pt({}).complete([FUNC_WRONG.encode()]) == 0.0


@pytest.mark.asyncio
async def test_public_unfenced_under_generic_scores_as_evaluator():
    # No `or generation` fallback: unfenced output under the fenced style yields
    # "" (empty code), matching the evaluator, so it is penalized, not run as code.
    pot = LCBPublicTestPotential(public_eval_sample=PUB_FUNC, extraction_style="generic")
    assert await pot.complete([FUNC_GOOD.encode()]) < 0  # FUNC_GOOD has no fences -> empty -> fails


def test_runtime_timeout_not_spoofable_by_exception_name():
    # _classify must use the exception type, not a repr substring, so generated
    # code cannot dodge the kill by raising an exception that name-drops timeout.
    from genlm.eval.domains.livecodebench.runtime_execution import _classify, RUNTIME, TIMEOUT, TimeoutException
    assert _classify(ValueError("TimeoutException-ish message")) == RUNTIME
    assert _classify(TimeoutException()) == TIMEOUT


def test_public_run_public_tests_reports_per_test_results():
    fb = _pt().run_public_tests(FUNC_HALF)
    assert fb.n_public == 2 and fb.n_passed == 1 and fb.n_failed == 1
    assert not fb.all_passed and fb.pass_fraction == 0.5
    passed = [r.passed for r in fb.results]
    assert passed == [True, False]
    failing = next(r for r in fb.results if not r.passed)
    assert failing.input == "10" and failing.expected == "20" and failing.got == "11"


def test_public_feedback_summary_lists_failing_cases():
    fb = _pt().run_public_tests(FUNC_WRONG)
    summary = fb.summary()
    assert "Passed 0/2 public tests" in summary
    assert "Input:" in summary and "Expected output:" in summary
    assert PublicTestFeedback(n_public=2, n_passed=2).summary() == "All 2 public tests passed."
    assert PublicTestFeedback(n_public=0, n_passed=0).summary() == "No public tests were available."


def test_public_caches_per_code():
    pot = _pt()
    fb1 = pot.run_public_tests(FUNC_WRONG)
    fb2 = pot.run_public_tests(FUNC_WRONG)
    assert fb1 is fb2  # second call served from cache


def test_public_coerce_adopts_vocab():
    from types import SimpleNamespace
    pot = _pt(penalty_per_failed=3.0, min_score=-7.0)
    pot2 = pot.coerce(SimpleNamespace(vocab=[b"x"]))
    assert list(pot2.vocab) == [b"x"]
    assert pot2.penalty_per_failed == 3.0 and pot2.min_score == -7.0


# --------------------------- repair prompt --------------------------- #

class _FakeTok:
    def __init__(self):
        self.last_text = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "<BOS>" + messages[-1]["content"]

    def encode(self, text, add_special_tokens=True):
        self.last_text = text
        return [1, 2, 3]


def _instance():
    return LiveCodeBenchInstance(instance_id="q", question_content="Double the number.",
                                 starter_code="", eval_sample={"input_output": "{}"})


def test_repair_question_content_includes_attempt_and_feedback():
    fb = _pt().run_public_tests(FUNC_WRONG)
    q = repair_question_content("Double it.", FUNC_WRONG, fb)
    assert "Double it." in q and "previous attempt" in q.lower()
    assert "return x + 1" in q and "Fix the program" in q
    assert "Passed 0/2 public tests" in q


def test_format_repair_prompt_returns_tokens_with_feedback():
    fb = _pt().run_public_tests(FUNC_WRONG)
    tok = _FakeTok()
    prev_gen = f"```python\n{FUNC_WRONG}```"
    ids = format_repair_prompt(tok, _instance(), prev_gen, fb, style="generic")
    assert ids == [1, 2, 3]
    assert "return x + 1" in tok.last_text          # previous code spliced in
    assert "public tests" in tok.last_text.lower()  # feedback spliced in


# ----------------------- extraction / decoding ----------------------- #

@pytest.mark.parametrize("output, style, expected", [
    ("```python\nprint(1)\n", "generic", "print(1)\n"),     # open block -> judge it
    ("```python\nprint(1)\n```\n", "generic", ""),          # closed block -> defer to complete()
    ("no code yet", "generic", ""),                         # no fence
    ("\n  print(1)\n", "genericbase", "print(1)"),          # whole output, stripped
])
def test_extract_code_prefix(output, style, expected):
    assert extract_code_prefix(output, style) == expected


@pytest.mark.asyncio
async def test_runtime_genericbase_leading_whitespace_prefix_not_killed():
    # A leading blank line / indentation must not make prefix() disagree with
    # complete() (which strips): both extract the same stripped code.
    pot = _rt(PUB_FUNC)
    assert await pot.prefix([("\n" + FUNC_GOOD).encode()]) == 0.0
    assert await pot.complete([("\n" + FUNC_GOOD).encode()]) == 0.0


@pytest.mark.asyncio
async def test_runtime_fenced_closed_block_defers_to_complete():
    # A closed fenced block is not judged at prefix (a later block may replace it),
    # so a broken first block can't kill a generation whose real answer comes later.
    pot = LCBRuntimeNoErrorPotential(public_eval_sample=PUB_STDIN, timeout_seconds=6.0)
    closed = "```python\nimport nonexistent_module_xyz\n```\nLet me retry:\n"
    assert await pot.prefix([closed.encode()]) == 0.0  # block 1 closed -> defer, not -inf
    # While that same block is still open, a real error is still caught early.
    open_block = "```python\nimport nonexistent_module_xyz\n"
    assert await pot.prefix([open_block.encode()]) == float("-inf")


@pytest.mark.parametrize("ctx, expected", [
    ("hi", "hi"),
    (b"hi", "hi"),
    ([104, 105], "hi"),
    ([b"hi", 32, b"there"], "hi there"),
    ([], ""),
])
def test_decode_context_shapes(ctx, expected):
    assert decode_context(ctx) == expected
