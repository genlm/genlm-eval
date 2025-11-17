import re
from pathlib import Path
from dataclasses import dataclass
from typing import List
import pytest

from genlm.eval.domains.goal_inference import (
    GoalInferenceDataset,
    GoalInferenceInstance,
    GoalInferenceEvaluator,
    GoalInferenceVALPotential,
)

# ---------------------------------------------------------------------
# Fixtures: Blocksworld domain
# ---------------------------------------------------------------------


@pytest.fixture
def bw_domain_text() -> str:
    with open("assets/goal_inference/pddl_domains/blocksworld.pddl") as f:
        return f.read()


@pytest.fixture
def bw_problem_text() -> str:
    return """(define (problem on_table_to_stack_1)
(:domain blocksworld)
(:requirements :strips)
(:objects b1)
(:init (arm-empty) (clear b1) (on-table b1))
(:goal (and (arm-empty) (clear b1) (on-table b1)))
)"""


@pytest.fixture
def nl_goal_text() -> str:
    return "Place b3 on b2, keep b1 on the table, and ensure the arm is empty."


@pytest.fixture
def dev_items(nl_goal_text, bw_problem_text) -> List[dict]:
    return [
        {
            "instance_id": 0,
            "nl_goal": nl_goal_text,
            "problem_text": bw_problem_text,
            "domain_name": "blocksworld",
        }
    ]


@pytest.fixture
def dataset(dev_items):
    return GoalInferenceDataset(dev_items)


@pytest.fixture
def evaluator():
    return GoalInferenceEvaluator()


# ---------------------------------------------------------------------
# Helpers (local)
# ---------------------------------------------------------------------


def _goal_mask(problem_text: str) -> str:
    i = problem_text.find("(:goal")
    if i == -1:
        return None
    prefix_before_goal = problem_text[:i]
    goal_suffix = "(:goal (and [BLANK]))\n)"
    return prefix_before_goal + goal_suffix


def _goal_prefix(problem_text: str) -> str:
    m = re.search(r"\(:goal\s*\(and", problem_text)
    if not m:
        return None
    return problem_text[: m.end()]


# ---------------------------------------------------------------------
# Dataset basic tests
# ---------------------------------------------------------------------


def test_goal_dataset_iter(dataset):
    assert dataset.schema is GoalInferenceInstance
    inst = next(iter(dataset))
    assert isinstance(inst, GoalInferenceInstance)
    assert inst.prefix_pddl == _goal_prefix(inst.problem_text)
    assert inst.masked_pddl == _goal_mask(inst.problem_text)


def test_goal_dataset_two_instances(dev_items):
    ds2 = GoalInferenceDataset(dev_items + [dev_items[0] | {"instance_id": 1}])
    it = list(iter(ds2))
    assert len(it) == 2
    assert it[0].instance_id == 0 and it[1].instance_id == 1


# ---------------------------------------------------------------------
# Evaluator tests
# ---------------------------------------------------------------------


def _stub_planetarium_equiv(monkeypatch):
    monkeypatch.setattr(
        "planetarium.evaluate",
        lambda x, y: (True, True, x == y),
    )


def test_goal_evaluator_equiv_true(dataset, evaluator, monkeypatch):
    _stub_planetarium_equiv(monkeypatch)
    inst = next(iter(dataset))
    good_pred = "(arm-empty) (clear b1) (on-table b1"
    result = evaluator.evaluate_sample(inst, good_pred)
    assert result.score == 1.0 and result.desc == "equiv"


def test_goal_evaluator_equiv_false(dataset, evaluator, monkeypatch):
    _stub_planetarium_equiv(monkeypatch)
    inst = next(iter(dataset))
    # Delete a literal
    bad_pred = "(arm-empty) (clear b1"
    result = evaluator.evaluate_sample(inst, bad_pred)
    assert result.score == 0.0 and result.desc == "not_equiv"


def test_goal_evaluator_planetarium_error(dataset, evaluator, monkeypatch):
    # Simulate parser error in planetarium
    def _boom(x, y):
        raise ValueError("parse error")

    monkeypatch.setattr(
        "planetarium.evaluate",
        _boom,
    )
    inst = next(iter(dataset))
    pred = "(arm-empty)"
    r = evaluator.evaluate_sample(inst, pred)
    assert r.score == 0.0 and r.desc == "planetarium_error"


# ----------------------------------------------------------------------
# GoalInferenceVALPotential (mock subprocess Fast-Downward & Validate)
# ----------------------------------------------------------------------


@dataclass
class _FakeProc:
    returncode: int
    stdout: str = ""
    stderr: str = ""


@pytest.fixture
def mock_subprocess_success(monkeypatch):
    def fake_run(cmd, shell, stdout, stderr, encoding, timeout=None):
        if "--plan-file" in cmd:
            return _FakeProc(returncode=0)
        if "Validate" in cmd or "validate" in cmd:
            return _FakeProc(returncode=0)
        return _FakeProc(returncode=0)

    monkeypatch.setattr(
        "genlm.eval.domains.goal_inference.goal_potential.subprocess.run", fake_run
    )
    return True


@pytest.fixture
def mock_subprocess_fail_val(monkeypatch):
    def fake_run(cmd, shell, stdout, stderr, encoding, timeout=None):
        if "--plan-file" in cmd:
            return _FakeProc(returncode=0)
        if "Validate" in cmd or "validate" in cmd:
            return _FakeProc(returncode=1, stderr="plan invalid")
        return _FakeProc(returncode=0)

    monkeypatch.setattr(
        "genlm.eval.domains.goal_inference.goal_potential.subprocess.run", fake_run
    )
    return True


@pytest.fixture
def exp_potential(bw_domain_text, bw_problem_text, mock_subprocess_success):
    return GoalInferenceVALPotential(
        domain_pddl_text=bw_domain_text,
        problem_pddl_text=bw_problem_text,
        fast_downward_cmd="./fast-downward.sif",
        val_cmd="Validate",
        cache_root=Path("tmp_goal_cache"),
        verbosity=0,
    )


@pytest.mark.asyncio
async def test_exp_prefix_success(exp_potential):
    # Partial string
    ctx = b"(on b3 b2"
    assert await exp_potential.prefix(ctx) == 0.0


@pytest.mark.asyncio
async def test_exp_complete_success(exp_potential):
    # Complete adds an extra ')' internally
    ctx = b"(on b3 b2"
    assert await exp_potential.complete(ctx) == 0.0


@pytest.mark.asyncio
async def test_exp_complete_failure(
    bw_domain_text, bw_problem_text, mock_subprocess_fail_val
):
    pot = GoalInferenceVALPotential(
        domain_pddl_text=bw_domain_text,
        problem_pddl_text=bw_problem_text,
        fast_downward_cmd="./fast-downward.sif",
        val_cmd="Validate",
        cache_root=Path("tmp_goal_cache"),
        verbosity=0,
    )
    assert await pot.complete(b"(on b3 b2") == float("-inf")
