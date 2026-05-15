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
    assert items[0].prompt == "P1"
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

    # Trailing newline but syntactically incomplete -> return 0.0
    s2 = "def foo(\n"
    score2 = await pot.prefix([s2.encode()])
    assert score2 == float("-inf")

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
    bad_syntax = "def foo(\n"                  # -> -inf, syntax_error=True
    good = "def foo():\n    return 42\n"       # -> 0.0,  syntax_error=False

    s1 = await pot.prefix([bad_syntax.encode()])
    assert s1 == float("-inf") and pot.last_was_syntax_error is True
    s2 = await pot.prefix([good.encode()])
    assert s2 == 0.0 and pot.last_was_syntax_error is False
    assert pot.cache_hits == 0 and pot.cache_misses == 2

    # Replay: must hit cache and restore the per-call flag correctly.
    s3 = await pot.prefix([bad_syntax.encode()])
    assert s3 == float("-inf") and pot.last_was_syntax_error is True
    s4 = await pot.prefix([good.encode()])
    assert s4 == 0.0 and pot.last_was_syntax_error is False
    assert pot.cache_hits == 2 and pot.cache_misses == 2


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
