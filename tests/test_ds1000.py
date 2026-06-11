import textwrap
from types import SimpleNamespace
import pytest

from genlm.eval.domains.ds1000 import (
    DS1000Evaluator,
    DS1000Dataset,
    DS1000Instance,
    DS1000RuntimeNoErrorPotential,
    _postprocess_code,
)

# ------------------------------ #
# Helpers                        #
# ------------------------------ #

def make_instance(code_context: str, prompt: str = "", meta=None):
    return SimpleNamespace(code_context=code_context, prompt=prompt, metadata=meta or {})

def harness_expect_foo_eq_42() -> str:
    # Loads the model's code into a namespace and checks that foo() returns 42.
    return (
        "def test_execution(solution: str):\n"
        "    ns = {}\n"
        "    exec(solution, ns, ns)\n"
        "    assert callable(ns.get('foo')), 'missing foo()'\n"
        "    assert ns['foo']() == 42, 'wrong answer'\n"
    )

def harness_expect_foo_eq_43() -> str:
    # Same as above but expects 43 for a failure.
    return (
        "def test_execution(solution: str):\n"
        "    ns = {}\n"
        "    exec(solution, ns, ns)\n"
        "    assert callable(ns.get('foo')), 'missing foo()'\n"
        "    assert ns['foo']() == 43, 'wrong answer'\n"
    )


@pytest.fixture
def evaluator():
    return DS1000Evaluator(timeout_seconds=0.2)

# ------------------------------ #
# DS1000Dataset                  #
# ------------------------------ #

def test_dataset():
    rows = [
        {
            "prompt": "P1 ",
            "code_context": "C1 ",
            "metadata": {"library": "pandas", "perturbation_type": "original"},
            "reference_code": "print('ok')",
        },
        {
            "prompt": "P2",
            "code_context": "C2",
            "metadata": {"library": "numpy", "perturbation_type": "rename"},
        },
    ]
    ds = DS1000Dataset(rows)
    assert len(ds) == 2
    assert ds.schema is DS1000Instance
    items = list(iter(ds))
    assert items[0].prompt == "P1 "  # prompt is kept verbatim
    assert items[0].code_context == "C1"
    assert items[0].metadata["library"] == "pandas"
    assert items[0].reference_code == "print('ok')"
    assert items[0].instance_id == 0
    assert items[1].instance_id == 1

# ------------------------------ #
# DS1000Evaluator                #
# ------------------------------ #

def test_postprocess_code_strips_fences_end_solution_and_html(evaluator):
    src = textwrap.dedent(
        """
        ```python
        x = 1
        y = 2
        ```
        END SOLUTION
        <code>
        </code>
        """
    )
    out = _postprocess_code(src)
    assert "x = 1" in out and "y = 2" in out
    assert "```" not in out
    assert "END SOLUTION" not in out
    assert "<code>" not in out


@pytest.mark.parametrize(
    "code,expected",
    [
        ("result = 5", True),
        ("result: int = 3", True),
        ("x = 1\n# result\nx", False),
        ("resuLT = 1  # different name", False),
    ],
)
def test_assigns_result_detection(evaluator, code, expected):
    assert evaluator.assigns_result(code) is expected


def test_evaluate_sample_truncates_desc(evaluator):
    instance = make_instance(harness_expect_foo_eq_42())
    evaluator.max_log_chars = 10
    long_solution = "def foo():\n    return 42\n" + ("#" * 100)
    res = evaluator.evaluate_sample(instance, long_solution)
    assert len(res.desc) <= 10 + len("\n...[truncated]")
    assert res.score == 1.0


def test_evaluate_sample_fail_when_harness_fails(evaluator):
    instance = make_instance(harness_expect_foo_eq_43())
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0
    assert ("fail" in res.desc.lower()) or ("wrong" in res.desc.lower()) or res.desc.strip().startswith("def foo()")


def test_run_in_subprocess_pass_and_fail_markers(evaluator):
    script_ok = 'print("<<<DS1000_PASS>>>")'
    ok, rc, out, err = evaluator._run_in_subprocess(script_ok)
    assert ok and rc == 0 and "<<<DS1000_PASS>>>" in out

    script_mixed = (
        'import sys; sys.stdout.write("<<<DS1000_PASS>>>\\n"); '
        'sys.stderr.write("<<<DS1000_FAIL>>> something\\n")'
    )
    ok, rc, out, err = evaluator._run_in_subprocess(script_mixed)
    assert not ok and rc == 0 and "<<<DS1000_FAIL>>>" in err

    script_bad_rc = 'import sys; print("<<<DS1000_PASS>>>"); sys.exit(3)'
    ok, rc, out, err = evaluator._run_in_subprocess(script_bad_rc)
    assert not ok and rc == 3


def test_run_in_subprocess_uses_extra_env(evaluator, monkeypatch):
    evaluator.extra_env = {"FOO_TEST_ENV": "BAR123"}
    script = (
        'import os; '
        'assert os.environ.get("FOO_TEST_ENV")=="BAR123"; '
        'print("<<<DS1000_PASS>>>")'
    )
    ok, rc, out, err = evaluator._run_in_subprocess(script)
    assert ok and rc == 0

# ------------------------------ #
# DS1000RuntimeNoErrorPotential  #
# ------------------------------ #

@pytest.mark.asyncio
async def test_runtime_potential_ok_on_assertion_failure_treated_as_ok():
    pot = DS1000RuntimeNoErrorPotential(code_context=harness_expect_foo_eq_43(), timeout_seconds=0.5)
    # wrong answer -> treated as OK by the potential
    solution = "def foo():\n    return 42\n\n"
    ctx = [solution.encode()]
    score = await pot.prefix(ctx)
    assert score == 0.0  # OK case


@pytest.mark.asyncio
async def test_runtime_potential_bad_on_exception_in_harness_call():
    pot = DS1000RuntimeNoErrorPotential(code_context=harness_expect_foo_eq_42(), timeout_seconds=0.5)
    # Calling foo raises -> -inf
    solution = "def foo():\n    raise RuntimeError('boom')\n\n"
    ctx = [solution.encode()]
    score = await pot.complete(ctx)
    assert score == float("-inf")


@pytest.mark.asyncio
async def test_runtime_potential_timeout(monkeypatch):
    # Harness with infinite loop -> timeout -> -inf
    code_context = (
        "def test_execution(solution):\n"
        "    while True:\n"
        "        pass\n"
    )
    pot = DS1000RuntimeNoErrorPotential(code_context=code_context, timeout_seconds=0.2)
    solution = "x = 1\n\n"
    score = await pot.complete([solution.encode()])
    assert score == float("-inf")


@pytest.mark.asyncio
async def test_runtime_potential_gating_requires_trailing_newline_and_parsable():
    pot = DS1000RuntimeNoErrorPotential(code_context=harness_expect_foo_eq_42(), timeout_seconds=0.5)

    # Missing trailing newline -> return 0.0
    s1 = "def foo():\n    return 42"  # no trailing \n
    score1 = await pot.prefix([s1.encode()])
    assert score1 == 0.0

    # Trailing newline but syntactically incomplete: a continuation can still
    # close the paren, so judgment is deferred (0.0), not -inf.
    s2 = "def foo(\n"
    score2 = await pot.prefix([s2.encode()])
    assert score2 == 0.0

    # Unfixable syntax error -> -inf at the prefix already.
    s2b = "definitely not valid python !\n"
    score2b = await pot.prefix([s2b.encode()])
    assert score2b == float("-inf")
    assert pot.last_was_syntax_error is True

    # Proper code -> executes harness -> OK
    s3 = "def foo():\n    return 42\n"
    score3 = await pot.prefix([s3.encode()])
    assert score3 == 0.0


def test_runtime_potential_coerce_adopts_vocab():
    other = SimpleNamespace(vocab=[b"a", b"b"])
    pot = DS1000RuntimeNoErrorPotential(code_context="")
    pot2 = pot.coerce(other)
    assert list(pot2.vocab) == [b"a", b"b"]


@pytest.mark.asyncio
async def test_runtime_potential_cache_replays_value_and_syntax_flag():
    pot = DS1000RuntimeNoErrorPotential(
        code_context=harness_expect_foo_eq_42(), timeout_seconds=0.5
    )
    bad_syntax = "definitely not valid python !\n"  # -> -inf, syntax_error=True (host-side)
    good = "def foo():\n    return 42\n"       # -> 0.0,  syntax_error=False

    s1 = await pot.prefix([bad_syntax.encode()])
    assert s1 == float("-inf") and pot.last_was_syntax_error is True
    s2 = await pot.prefix([good.encode()])
    assert s2 == 0.0 and pot.last_was_syntax_error is False
    # broken syntax is rejected host-side without a subprocess; only the good
    # prefix reaches the scorer.
    assert pot.cache_hits == 0 and pot.cache_misses == 1

    # Replay: must hit cache and restore the per-call flag correctly.
    s3 = await pot.prefix([bad_syntax.encode()])
    assert s3 == float("-inf") and pot.last_was_syntax_error is True
    s4 = await pot.prefix([good.encode()])
    assert s4 == 0.0 and pot.last_was_syntax_error is False
    assert pot.cache_hits == 1 and pot.cache_misses == 1

    # complete() still scores broken syntax through the subprocess path.
    s5 = await pot.complete([bad_syntax.encode()])
    assert s5 == float("-inf") and pot.last_was_syntax_error is True


@pytest.mark.asyncio
async def test_runtime_potential_timeout_is_not_cached():
    code_context = (
        "def test_execution(solution):\n"
        "    while True:\n"
        "        pass\n"
    )
    pot = DS1000RuntimeNoErrorPotential(code_context=code_context, timeout_seconds=0.2)
    score = await pot.complete([b"x = 1\n\n"])
    assert score == float("-inf")
    assert len(pot._score_cache) == 0


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("hello", "hello"),
        (b"hello", "hello"),
        ([104, 105], "hi"),                   # List[int] byte ids
        ([b"hi", 32, b"there"], "hi there"),  # mixed list
        ([], ""),
    ],
)
def test_runtime_potential_bytes_to_str_input_shapes(inp, expected):
    pot = DS1000RuntimeNoErrorPotential(code_context="")
    assert pot._bytes_to_str(inp) == expected


# ------------------------------ #
# Additional evaluator scenarios #
# ------------------------------ #

def test_missing_test_execution_via_build_harness_script(evaluator):
    code_context = "def not_test_execution(solution):\n    pass\n"
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0


def test_harness_top_level_exec_error(evaluator):
    code_context = (
        "raise RuntimeError('boom at import')\n"
        "def test_execution(solution):\n"
        "    return None\n"
    )
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0


def test_harness_timeout_in_test_execution(evaluator):
    code_context = (
        "def test_execution(solution):\n"
        "    while True:\n"
        "        pass\n"
    )
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0


# ----------------------------------------- #
# Prefix semantics on a DS-1000-like harness #
# ----------------------------------------- #

# Mirrors the structure of real DS-1000 code contexts: an exec_context with an
# [insert] placeholder and a tail that references a variable the solution must
# define, and a test_execution that extracts
# test_env["result"] after execution.
DS1000_LIKE_CONTEXT = '''
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        return sorted(data)

    def define_test_input(test_case_id):
        return [3, 1, 2]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
data = test_input
[insert]
result = answer_list
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
'''


@pytest.fixture
def ds1000_like_potential():
    return DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0
    )


@pytest.mark.asyncio
async def test_prefix_survives_partial_multiline_solution(ds1000_like_potential):
    # First line of a correct two-line solution: does not define `answer_list`
    # yet, so the exec_context tail / result extraction would raise
    # NameError/KeyError. The prefix must NOT be killed.
    score = await ds1000_like_potential.prefix([b"tmp = list(data)\n"])
    assert score == 0.0
    # ... and the full solution passes prefix and complete.
    full = "tmp = list(data)\nanswer_list = sorted(tmp)\n"
    assert await ds1000_like_potential.prefix([full.encode()]) == 0.0
    assert await ds1000_like_potential.complete([full.encode()]) == 0.0


@pytest.mark.asyncio
async def test_prefix_defers_open_block_then_checks_when_closed(ds1000_like_potential):
    pot = ds1000_like_potential
    # Open for-loop: syntactically incomplete -> defer.
    assert await pot.prefix([b"answer_list = []\nfor v in sorted(data):\n"]) == 0.0
    # Body present but loop still extendable -> trailing compound is skipped.
    s = b"answer_list = []\nfor v in sorted(data):\n    answer_list.append(v)\n"
    assert await pot.prefix([s]) == 0.0
    # Once a top-level statement follows, the loop is closed and executes.
    s_err = (
        b"answer_list = []\n"
        b"for v in sorted(data):\n"
        b"    answer_list.append(undefined_name)\n"
        b"x = 1\n"
    )
    assert await pot.prefix([s_err]) == float("-inf")


@pytest.mark.asyncio
async def test_terminal_marker_applies_strict_check(ds1000_like_potential):
    pot = ds1000_like_potential
    # After </code> the postprocessed solution is final: a missing answer
    # variable is now a definitive failure, not a deferrable prefix state.
    assert await pot.prefix([b"tmp = list(data)\n</code>\n"]) == float("-inf")
    assert await pot.prefix([b"answer_list = sorted(data)\n</code>\n"]) == 0.0
    assert await pot.prefix([b"answer_list = sorted(data)\nEND SOLUTION\n"]) == 0.0
    assert await pot.prefix([b"tmp = 1\nEND SOLUTION\n"]) == float("-inf")
    # A terminally-empty generation can never become a solution.
    assert await pot.prefix([b"</code>\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_hung_prefix_extensions_skip_subprocess(ds1000_like_potential):
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=4.0
    )
    calls = []
    orig = pot._run_script

    async def counting(script):
        calls.append(1)
        return await orig(script)

    pot._run_script = counting
    hang = b"while True:\n    pass\nx = 1\n"
    assert await pot.prefix([hang]) == 0.0
    n = len(calls)
    assert n >= 1
    # Extending a hung prefix re-runs the same leading statements: defer
    # without launching another subprocess.
    assert await pot.prefix([hang + b"y = 2\n"]) == 0.0
    assert len(calls) == n


@pytest.fixture
def fast_forkserver_env(monkeypatch):
    # Skip heavy pre-imports so worker warmup is instant in tests.
    monkeypatch.setenv("DS1000_FORKSERVER_PRELOAD", "")


@pytest.mark.asyncio
async def test_forkserver_backend_matches_subprocess_verdicts(fast_forkserver_env):
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    assert await pot.prefix([b"tmp = list(data)\n"]) == 0.0
    full = b"tmp = list(data)\nanswer_list = sorted(tmp)\n"
    assert await pot.prefix([full]) == 0.0
    assert await pot.complete([full]) == 0.0
    assert await pot.complete([b"tmp = list(data)\n"]) == float("-inf")
    assert await pot.prefix([b"import nonexistent_module_xyz\n"]) == float("-inf")


def test_session_scripts_are_valid_python():
    # Template escaping bugs silently degrade to the (slow) fallback path,
    # so the generated session scripts must always parse.
    import ast as _ast

    for ctx in (DS1000_LIKE_CONTEXT, FUNCTION_BODY_CONTEXT, MATPLOTLIB_LIKE_CONTEXT):
        pot = DS1000RuntimeNoErrorPotential(code_context=ctx)
        _ast.parse(pot._session_setup_script())
        _ast.parse(pot._session_body_script("x = 1\n", "<OK>", "<BAD>", "<SYN>"))
        _ast.parse(pot._prefix_script("x = 1\n", "<OK>", "<BAD>", "<SYN>"))
        _ast.parse(pot._complete_script("x = 1\n", "<OK>", "<BAD>", "<SYN>"))


@pytest.mark.asyncio
async def test_session_function_body_and_errors(fast_forkserver_env):
    # Function-body solutions through the warm-session path: the head only
    # parses combined with the solution, so the body script must reproduce
    # the line-attribution logic.
    pot = DS1000RuntimeNoErrorPotential(
        code_context=FUNCTION_BODY_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    assert await pot.prefix([b"    return sorted(data)\n"]) == 0.0
    assert await pot.prefix([b"    definitely not valid python !\n"]) == float(
        "-inf"
    )


@pytest.mark.asyncio
async def test_session_setup_failure_falls_back(fast_forkserver_env):
    # FILE_SETUP_CONTEXT's generate_test_case fails outside test_execution
    # (missing file), so the session setup fails; the worker must serve the
    # request via the stateless fallback, which defers (0.0).
    pot = DS1000RuntimeNoErrorPotential(
        code_context=FILE_SETUP_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    assert await pot.prefix([b"answer_list = sorted(data)\n"]) == 0.0


@pytest.mark.asyncio
async def test_sessions_disabled_knob(fast_forkserver_env, monkeypatch):
    monkeypatch.setenv("DS1000_FORKSERVER_SESSIONS", "0")
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    assert await pot.prefix([b"tmp = list(data)\n"]) == 0.0
    assert await pot.prefix([b"import nonexistent_module_xyz\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_forkserver_falls_back_to_subprocess(fast_forkserver_env, monkeypatch):
    from genlm.eval.domains.ds1000 import forkserver as fs

    # Unstartable worker: the potential must still answer via subprocess.
    monkeypatch.setattr(fs, "_WORKER_PATH", "/nonexistent/worker.py")
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    full = b"tmp = list(data)\nanswer_list = sorted(tmp)\n"
    assert await pot.complete([full]) == 0.0
    assert await pot.complete([b"tmp = list(data)\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_forkserver_worker_death_falls_back(fast_forkserver_env):
    from genlm.eval.domains.ds1000.forkserver import shared_executor

    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0, use_forkserver=True
    )
    full = b"tmp = list(data)\nanswer_list = sorted(tmp)\n"
    assert await pot.complete([full]) == 0.0
    # Kill the shared worker mid-session: subsequent calls must not crash
    # (restart or subprocess fallback) and verdicts stay correct.
    executor = shared_executor(pot.python_executable, pot.extra_env)
    executor.kill()
    assert await pot.complete([b"tmp = list(data)\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_prefix_defers_trailing_backslash_continuation(ds1000_like_potential):
    pot = ds1000_like_potential
    # A trailing line-continuation can be completed by the next line.
    assert await pot.prefix([b"answer_list = \\\n"]) == 0.0
    full = b"answer_list = \\\n    sorted(data)\n"
    assert await pot.prefix([full]) == 0.0
    assert await pot.complete([full]) == 0.0


# The harness writes a file inside test_execution before its test loop;
# generate_test_case reads it. Prefix checks cannot replicate that setup, so
# environment failures while building the test inputs must defer, not kill.
FILE_SETUP_CONTEXT = '''
def generate_test_case(test_case_id):
    with open("data.txt") as fh:
        data = [int(x) for x in fh.read().split()]
    return data, sorted(data)


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
data = test_input
[insert]
result = answer_list
"""


def test_execution(solution: str):
    with open("data.txt", "w") as fh:
        fh.write("3 1 2")
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
'''


@pytest.mark.asyncio
async def test_prefix_defers_harness_side_setup_failures():
    pot = DS1000RuntimeNoErrorPotential(
        code_context=FILE_SETUP_CONTEXT, timeout_seconds=10.0
    )
    sol = b"answer_list = sorted(data)\n"
    assert await pot.prefix([sol]) == 0.0
    assert await pot.complete([sol]) == 0.0


@pytest.mark.asyncio
async def test_prefix_timeout_defers_complete_timeout_kills():
    # A slow prefix may still have a valid continuation: prefix defers (0.0),
    # only complete() treats the timeout as fatal.
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=5.0
    )
    hang = b"while True:\n    pass\nx = 1\n"
    assert await pot.prefix([hang]) == 0.0
    assert await pot.complete([hang]) == float("-inf")


@pytest.mark.asyncio
async def test_prefix_kills_real_runtime_errors(ds1000_like_potential):
    # Genuine error in already-complete statements: no continuation can fix it.
    score = await ds1000_like_potential.prefix(
        [b"import nonexistent_module_xyz\n"]
    )
    assert score == float("-inf")


@pytest.mark.asyncio
async def test_prefix_does_not_execute_trailing_compound(ds1000_like_potential):
    # The trailing loop raises if executed, but the generation may still
    # extend its body, so the prefix must not be killed.
    s = b"for v in data:\n    raise ValueError(v)\n"
    assert await ds1000_like_potential.prefix([s]) == 0.0


@pytest.mark.asyncio
async def test_complete_still_requires_answer_variable(ds1000_like_potential):
    # complete() keeps strict harness semantics: missing `answer_list`
    # (and hence `result`) is an error.
    assert await ds1000_like_potential.complete([b"tmp = list(data)\n"]) == float(
        "-inf"
    )


@pytest.mark.asyncio
async def test_strict_prefix_runs_full_harness_on_prefixes(ds1000_like_potential):
    pot = DS1000RuntimeNoErrorPotential(
        code_context=DS1000_LIKE_CONTEXT, timeout_seconds=10.0, strict_prefix=True
    )
    # The full harness kills the partial solution (NameError on the
    # exec_context tail).
    assert await pot.prefix([b"tmp = list(data)\n"]) == float("-inf")


# Mirrors the matplotlib problems: exec_context sets up the "figure" state and
# the solution mutates it; exec_test then inspects state the solution may not
# have produced yet (a half-drawn plot has no legend -> None subscription).
MATPLOTLIB_LIKE_CONTEXT = '''
def skip_plt_cmds(l):
    return all(p not in l for p in ["plt.show()", "plt.clf()", "plt.close()", "savefig"])


def generate_test_case(test_case_id):
    return None, None


def exec_test(result, ans):
    leg = result.get("legend")
    assert leg[0] == "x-y"  # TypeError on half-drawn plot (legend is None)
    return 1


exec_context = r"""
_state = {}


def plot(x):
    _state["plotted"] = True


def legend(labels):
    _state["legend"] = list(labels)


[insert]
result = _state
"""


def test_execution(solution: str):
    solution = "\\n".join(filter(skip_plt_cmds, solution.split("\\n")))
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
'''


# Mirrors the function-completion problems: the solution is an
# *indented* function body inserted inside `def f(...):`, so it does not parse
# standalone, and the exec_context tail calls the function.
FUNCTION_BODY_CONTEXT = '''
import copy


def generate_test_case(test_case_id):
    def generate_ans(data):
        return sorted(data)

    def define_test_input(test_case_id):
        return [3, 1, 2]

    test_input = define_test_input(test_case_id)
    expected_result = generate_ans(copy.deepcopy(test_input))
    return test_input, expected_result


def exec_test(result, ans):
    assert result == ans
    return 1


exec_context = r"""
data = test_input
def f(data):
[insert]
result = f(data)
"""


def test_execution(solution: str):
    code = exec_context.replace("[insert]", solution)
    for i in range(1):
        test_input, expected_result = generate_test_case(i + 1)
        test_env = {"test_input": test_input}
        exec(code, test_env)
        assert exec_test(test_env["result"], expected_result)
'''


@pytest.mark.asyncio
async def test_function_body_solutions_parse_with_context():
    pot = DS1000RuntimeNoErrorPotential(
        code_context=FUNCTION_BODY_CONTEXT, timeout_seconds=10.0
    )
    # An indented body does not parse standalone but must NOT be killed:
    # neither at the prefix nor at complete().
    body = b"    return sorted(data)\n"
    assert await pot.prefix([body]) == 0.0
    assert await pot.complete([body]) == 0.0
    # Multi-line body, first line only -> prefix defers/passes.
    partial = b"    tmp = list(data)\n"
    assert await pot.prefix([partial]) == 0.0
    full = b"    tmp = list(data)\n    return sorted(tmp)\n"
    assert await pot.prefix([full]) == 0.0
    assert await pot.complete([full]) == 0.0
    # Garbage inside the body is still rejected.
    assert await pot.prefix([b"    definitely not valid python !\n"]) == float(
        "-inf"
    )
    # A body that errors at call time is caught at complete()
    # (the def itself is the trailing compound, so prefix cannot judge it).
    assert await pot.complete([b"    return undefined_name_xyz\n"]) == float("-inf")


@pytest.mark.asyncio
async def test_prefix_skips_figure_inspection_on_half_drawn_plot():
    # Mirrors the matplotlib failure mode: exec_test inspects the legend,
    # which does not exist yet after only the plot call. The prefix must not
    # run exec_test; complete() must.
    pot = DS1000RuntimeNoErrorPotential(
        code_context=MATPLOTLIB_LIKE_CONTEXT, timeout_seconds=10.0
    )
    assert await pot.prefix([b"plot([1, 2])\n"]) == 0.0
    full = b"plot([1, 2])\nlegend(['x-y'])\n"
    assert await pot.prefix([full]) == 0.0
    assert await pot.complete([full]) == 0.0
    # Without the legend the complete harness errors (TypeError on None).
    assert await pot.complete([b"plot([1, 2])\n"]) == float("-inf")


@pytest.mark.parametrize("context", [[], list(b"</code>")])
def test_complete_empty_short_circuits_subprocess(context, monkeypatch):
    import asyncio
    pot = DS1000RuntimeNoErrorPotential(code_context=harness_expect_foo_eq_42())
    called = []
    async def _spy(self, code):
        called.append(code)
        return 0.0
    monkeypatch.setattr(DS1000RuntimeNoErrorPotential, "_score_no_error", _spy)
    assert asyncio.run(pot.complete(context)) == float("-inf")
    assert called == []
