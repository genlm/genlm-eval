import pytest
from types import SimpleNamespace

from genlm.eval.domains.ds1000 import DS1000Evaluator 

# TODO extend tests

def make_instance(code_context: str):
    """Create a minimal instance."""
    return SimpleNamespace(code_context=code_context)


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


@pytest.fixture
def passing_instance():
    return make_instance(harness_expect_foo_eq_42())


@pytest.fixture
def failing_instance():
    return make_instance(harness_expect_foo_eq_43())


def test_evaluate_sample_pass(evaluator, passing_instance):
    solution = "def foo():\n    return 42\n"
    res = evaluator.evaluate_sample(passing_instance, solution)
    assert res.score == 1.0
    assert "pass" in res.desc.lower()


def test_evaluate_sample_fail(evaluator, failing_instance):
    solution = "def foo():\n    return 42\n"
    res = evaluator.evaluate_sample(failing_instance, solution)
    assert res.score == 0.0
    assert "fail" in res.desc.lower() or "wrong answer" in res.desc.lower()


def test_empty_solution(evaluator, passing_instance):
    res = evaluator.evaluate_sample(passing_instance, "   \n  ")
    assert res.score == 0.0
    assert "empty" in res.desc.lower()


def test_missing_test_execution(evaluator):
    code_context = "def not_test_execution(solution):\n    pass\n"
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0
    assert "missing" in res.desc.lower() or "fail" in res.desc.lower()


def test_harness_exec_error(evaluator):
    # error at module top-level triggers HARNESS_EXEC_ERROR
    code_context = (
        "raise RuntimeError('harness exploded')\n"
        "def test_execution(solution):\n"
        "    pass\n"
    )
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0
    assert "error" in res.desc.lower() or "fail" in res.desc.lower()


def test_timeout(evaluator):
    # Infinite loop inside test_execution -> evaluator should time out
    code_context = (
        "def test_execution(solution):\n"
        "    while True:\n"
        "        pass\n"
    )
    instance = make_instance(code_context)
    res = evaluator.evaluate_sample(instance, "def foo():\n    return 42\n")
    assert res.score == 0.0
    assert "timeout" in res.desc.lower()
