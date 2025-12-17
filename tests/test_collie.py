import pytest

from genlm.eval.domains.collie import (
    CollieDataset,
    CollieEvaluator,
    CollieInstance,
    default_prompt_formatter,
    CollieConstraintPotential,
)
from collie import constraints as cc
from genlm.eval import ModelOutput, ModelResponse

# ------------------------------ #
# Helpers                        #
# ------------------------------ #


def make_instance(
    prompt: str = "Generate a sentence with exactly 5 words.",
    example: str = "This is an example sentence.",
    targets=None,
    constraint_type: str = "test_c01",
    metadata=None,
    constraint=None,
):
    """Create a test CollieInstance with a real constraint object."""
    # Default constraint: word count == 5
    if constraint is None:
        constraint = cc.Constraint(
            input_level=cc.InputLevel(None),
            target_level=cc.TargetLevel("word"),
            transformation=cc.Count(),
            relation=cc.Relation("=="),
            reduction=cc.Reduction(None),
        )

    if targets is None:
        targets = 5

    return CollieInstance(
        prompt=prompt,
        example=example,
        targets=targets,
        metadata=metadata or {},
        constraint_type=constraint_type,
        constraint=constraint,
        instance_id=0,
    )


@pytest.fixture
def evaluator():
    return CollieEvaluator()


# ------------------------------ #
# CollieDataset                  #
# ------------------------------ #


def test_dataset():
    """Test basic dataset construction."""

    rows = [
        {
            "prompt": "Generate a sentence with exactly 5 words.",
            "example": "This is an example sentence.",
            "targets": 5,
            "metadata": {"source": "test"},
            "constraint_type": "test_c01",
            "constraint": cc.Constraint(
                cc.InputLevel(None),
                cc.TargetLevel("word"),
                cc.Count(),
                cc.Relation("=="),
                cc.Reduction(None),
            ),
        },
        {
            "prompt": "Generate a sentence with exactly 11 words.",
            "example": "The little group stayed back and talked in the empty chamber.",
            "targets": 11,
            "metadata": {"source": "test"},
            "constraint_type": "test_c02",
            "constraint": cc.Constraint(
                cc.InputLevel(None),
                cc.TargetLevel("word"),
                cc.Count(),
                cc.Relation("=="),
                cc.Reduction(None),
            ),
        },
    ]
    ds = CollieDataset(rows)
    assert len(ds) == 2
    assert ds.schema is CollieInstance
    items = list(iter(ds))
    assert items[0].prompt == "Generate a sentence with exactly 5 words."
    assert items[0].example == "This is an example sentence."
    assert items[0].targets == 5
    assert items[0].constraint_type == "test_c01"
    assert items[0].instance_id == 0
    assert items[1].instance_id == 1


def test_dataset_from_official():
    """Test loading from official Princeton-NLP repository."""
    # Load a small subset
    ds = CollieDataset.from_official(max_instances=5)
    assert len(ds) == 5

    # Check structure of first instance
    first = next(iter(ds))
    assert hasattr(first, "constraint")
    assert hasattr(first, "targets")
    assert hasattr(first, "example")
    assert hasattr(first, "prompt")
    assert hasattr(first, "metadata")
    assert hasattr(first.constraint, "check")


def test_dataset_filtering():
    """Test dataset filtering options."""
    from genlm.eval.domains.collie import RECOMMENDED_CONSTRAINT_TYPES

    # Test constraint type filtering
    ds = CollieDataset.from_official(
        constraint_types=["wiki_c01", "guten_c01"], max_instances=10
    )
    assert len(ds) <= 10
    for instance in ds:
        assert instance.constraint_type in ["wiki_c01", "guten_c01"]

    # Test example length filtering
    ds = CollieDataset.from_official(max_example_length=200, max_instances=10)
    assert len(ds) <= 10
    for instance in ds:
        assert len(instance.example) <= 200

    # Test prompt length filtering
    ds = CollieDataset.from_official(max_prompt_length=150, max_instances=10)
    assert len(ds) <= 10
    for instance in ds:
        assert len(instance.prompt) <= 150

    # Test combined filtering
    ds = CollieDataset.from_official(
        constraint_types=RECOMMENDED_CONSTRAINT_TYPES,
        max_example_length=300,
        max_prompt_length=200,
        max_instances=5,
    )
    assert len(ds) <= 5
    for instance in ds:
        assert instance.constraint_type in RECOMMENDED_CONSTRAINT_TYPES
        assert len(instance.example) <= 300
        assert len(instance.prompt) <= 200


def test_dataset_missing_fields():
    """Test dataset with missing fields uses defaults."""

    rows = [
        {
            "example": "Test",
            "constraint": cc.Constraint(
                cc.InputLevel(None),
                cc.TargetLevel("word"),
                cc.Count(),
                cc.Relation("=="),
                cc.Reduction(None),
            ),
        }
    ]
    ds = CollieDataset(rows)
    items = list(ds)
    assert items[0].prompt == ""
    assert items[0].metadata == {}


# ------------------------------ #
# CollieEvaluator                #
# ------------------------------ #


def test_evaluate_sample_empty_response(evaluator):
    instance = make_instance()
    result = evaluator.evaluate_sample(instance, "")
    assert result.score == 0.0
    assert "empty" in result.desc.lower()


def test_evaluate_sample_word_count(evaluator):
    """Test word count constraint."""

    # Create constraint: word count == 5
    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("word"),
        cc.Count(),
        cc.Relation("=="),
        cc.Reduction(None),
    )

    instance = make_instance(targets=5, constraint=constraint)

    # 5-word sentence
    result = evaluator.evaluate_sample(instance, "one two three four five")
    assert result.score == 1.0

    # 6-word sentence (should fail)
    result = evaluator.evaluate_sample(instance, "This sentence has six words total.")
    assert result.score == 0.0


def test_evaluate_sample_word_count_11(evaluator):
    """Test word count with 11 words."""

    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("word"),
        cc.Count(),
        cc.Relation("=="),
        cc.Reduction(None),
    )

    instance = make_instance(
        targets=11, constraint_type="ccnews_c05", constraint=constraint
    )

    # Valid: exactly 11 words
    result = evaluator.evaluate_sample(
        instance, "The little group stayed back and talked in the empty chamber."
    )
    assert result.score == 1.0

    # Invalid: 6 words
    result = evaluator.evaluate_sample(instance, "The group stayed in the chamber.")
    assert result.score == 0.0


def test_evaluate_sample_foreach(evaluator):
    """Test ForEach constraint: all words have exactly 2 characters."""

    constraint = cc.Constraint(
        cc.InputLevel("word"),
        cc.TargetLevel("character"),
        cc.ForEach(cc.Count()),
        cc.Relation("=="),
        cc.Reduction("all"),
    )

    instance = make_instance(
        targets=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        constraint_type="ccnews_c05",
        constraint=constraint,
    )

    # Valid: 14 words, all exactly 2 chars
    valid = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn"
    result = evaluator.evaluate_sample(instance, valid)
    assert result.score == 1.0

    # Invalid: last word has 3 chars
    invalid = "aa bb cc dd ee ff gg hh ii jj kk ll mm nnn"
    result = evaluator.evaluate_sample(instance, invalid)
    assert result.score == 0.0


def test_evaluate_sample_character_count(evaluator):
    """Test character count constraint."""

    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("character"),
        cc.Count(),
        cc.Relation("=="),
        cc.Reduction(None),
    )

    instance = make_instance(
        targets=91, constraint_type="guten_c04", constraint=constraint
    )

    # Exactly 91 characters
    response = "Here, the enemy would find it almost impossible to bring his heavy siege guns within range."
    assert len(response) == 91
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 1.0

    # Wrong character count
    result = evaluator.evaluate_sample(instance, "Short.")
    assert result.score == 0.0


def test_evaluate_sample_paragraph_count(evaluator):
    """Test paragraph count constraint."""

    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("paragraph"),
        cc.Count(),
        cc.Relation("=="),
        cc.Reduction(None),
    )

    instance = make_instance(
        targets=2, constraint_type="guten_c14", constraint=constraint
    )

    # Valid: 2 paragraphs
    response = "First paragraph.\n\nSecond paragraph."
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 1.0

    # Invalid: 1 paragraph
    response = "Single paragraph."
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 0.0


def test_evaluate_ensemble(evaluator):
    """Test ensemble evaluation."""

    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("word"),
        cc.Count(),
        cc.Relation("=="),
        cc.Reduction(None),
    )

    instance = make_instance(targets=6, constraint=constraint)

    ensemble_result = evaluator.evaluate_ensemble(
        instance,
        ModelOutput(
            responses=[
                ModelResponse(response="This is a six word sentence.", weight=0.6),
                ModelResponse(response="Only four words.", weight=0.4),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] == 0.6


def test_evaluator_requires_collie_library():
    """Test that evaluator requires collie-bench to be installed."""
    # This should work since collie-bench is installed
    evaluator = CollieEvaluator()
    assert evaluator is not None


# ------------------------------ #
# Prompt Formatter               #
# ------------------------------ #


def test_default_prompt_formatter():
    """Test the default prompt formatter."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    instance = make_instance()

    # Test without chat format
    token_ids = default_prompt_formatter(tokenizer, instance, use_chat_format=False)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert all(isinstance(t, int) for t in token_ids)


def test_evaluate_sample_unknown_target_format(evaluator):
    """Test evaluator with various target formats."""

    # Test with None target
    constraint = cc.Constraint(
        cc.InputLevel(None),
        cc.TargetLevel("word"),
        cc.Count(),
        cc.Relation(">="),
        cc.Reduction(None),
    )

    instance = make_instance(targets=None, constraint=constraint)
    result = evaluator.evaluate_sample(instance, "Some response text.")
    # Should not crash - exact behavior depends on Collie's handling of None
    assert result.score in [0.0, 1.0]


def test_evaluate_sample_with_metadata(evaluator):
    """Test that metadata is preserved in evaluation result."""
    instance = make_instance(metadata={"source": "test", "id": 123})
    result = evaluator.evaluate_sample(instance, "one two three four five")
    assert result.score == 1.0
    assert "constraint_type" in result.metadata
    assert result.metadata["constraint_type"] == "test_c01"


# ------------------------------ #
# CollieConstraintPotential      #
# ------------------------------ #


class TestCollieConstraintPotential:
    """Tests for CollieConstraintPotential."""

    @pytest.mark.asyncio
    async def test_word_count_constraint_partial(self):
        """Test word count constraint on partial text."""
        # Create a constraint: word count == 10
        constraint = cc.Constraint(
            cc.InputLevel(None),
            cc.TargetLevel("word"),
            cc.Count(),
            cc.Relation("=="),
            cc.Reduction(None),
        )

        potential = CollieConstraintPotential(constraint, targets=10, tolerance=1.5)

        # Test partial text with 5 words (should be OK - below target)
        partial_text = "This is a partial sentence".encode("utf-8")
        score = await potential.prefix(partial_text)
        assert score == 0.0, "Partial text below target should be accepted"

        # Test partial text with 10 words (should be OK - at target)
        at_target_text = "One two three four five six seven eight nine ten".encode(
            "utf-8"
        )
        score = await potential.prefix(at_target_text)
        assert score == 0.0, "Partial text at target should be accepted"

        # Test partial text with 20 words (should fail - way above target * tolerance)
        over_text = ("One two three four five six seven eight nine ten " * 2).encode(
            "utf-8"
        )
        score = await potential.prefix(over_text)
        assert score == float("-inf"), "Partial text way over target should be rejected"

    @pytest.mark.asyncio
    async def test_word_count_constraint_complete(self):
        """Test word count constraint on complete text."""
        constraint = cc.Constraint(
            cc.InputLevel(None),
            cc.TargetLevel("word"),
            cc.Count(),
            cc.Relation("=="),
            cc.Reduction(None),
        )

        potential = CollieConstraintPotential(constraint, targets=10)

        # Test complete text with exactly 10 words (should pass)
        correct_text = "One two three four five six seven eight nine ten".encode(
            "utf-8"
        )
        score = await potential.complete(correct_text)
        assert score == 0.0, "Complete text with correct word count should pass"

        # Test complete text with wrong count (should fail)
        wrong_text = "One two three four five".encode("utf-8")
        score = await potential.complete(wrong_text)
        assert score == float("-inf"), "Complete text with wrong word count should fail"

    @pytest.mark.asyncio
    async def test_character_count_constraint(self):
        """Test character count constraint."""
        constraint = cc.Constraint(
            cc.InputLevel(None),
            cc.TargetLevel("character"),
            cc.Count(),
            cc.Relation("=="),
            cc.Reduction(None),
        )

        potential = CollieConstraintPotential(constraint, targets=20, tolerance=1.5)

        # Test partial text with 10 characters (should be OK)
        partial_text = "1234567890".encode("utf-8")
        score = await potential.prefix(partial_text)
        assert score == 0.0, "Partial text below character target should be accepted"

        # Test complete text with exactly 20 characters (should pass)
        correct_text = "12345678901234567890".encode("utf-8")
        score = await potential.complete(correct_text)
        assert score == 0.0, "Complete text with correct character count should pass"

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test that empty text is handled correctly."""
        constraint = cc.Constraint(
            cc.InputLevel(None),
            cc.TargetLevel("word"),
            cc.Count(),
            cc.Relation("=="),
            cc.Reduction(None),
        )

        potential = CollieConstraintPotential(constraint, targets=10)

        # Empty text should be OK during partial generation
        empty_text = b""
        score = await potential.prefix(empty_text)
        assert score == 0.0, "Empty partial text should be accepted"

    @pytest.mark.asyncio
    async def test_foreach_constraint(self):
        """Test ForEach constraint (e.g., each word has exactly 2 characters)."""
        constraint = cc.Constraint(
            cc.InputLevel("word"),
            cc.TargetLevel("character"),
            cc.ForEach(cc.Count()),
            cc.Relation("=="),
            cc.Reduction("all"),
        )

        # Target: each word should have exactly 2 characters
        targets = [2, 2, 2, 2, 2]
        potential = CollieConstraintPotential(
            constraint, targets=targets, tolerance=1.5
        )

        # Test partial text with 2-char words (should be OK)
        partial_text = "aa bb cc".encode("utf-8")
        score = await potential.prefix(partial_text)
        assert score == 0.0, "Partial text with correct word lengths should be accepted"

        # Test complete text with all 2-char words (should pass)
        correct_text = "aa bb cc dd ee".encode("utf-8")
        score = await potential.complete(correct_text)
        assert score == 0.0, "Complete text with correct word lengths should pass"

        # Test complete text with wrong word length (should fail)
        wrong_text = "aa bb ccc dd ee".encode("utf-8")  # 'ccc' has 3 chars
        score = await potential.complete(wrong_text)
        assert score == float(
            "-inf"
        ), "Complete text with wrong word length should fail"
