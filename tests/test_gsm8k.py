import pytest
from types import SimpleNamespace

from genlm.eval.domains.gsm8k import (
    GSM8KInstance,
    GSM8KDataset,
    GSM8KEvaluator,
    extract_answer,
    extract_ground_truth,
    default_prompt_formatter,
    chain_of_thought_prompt_formatter,
    direct_answer_prompt_formatter,
    few_shot_prompt_formatter,
    FEW_SHOT_EXAMPLES,
)
from genlm.eval.core import ModelOutput, ModelResponse


# ------------------------------ #
# Helpers                        #
# ------------------------------ #


def make_instance(question: str = "", answer: str = "", meta=None):
    return SimpleNamespace(
        question=question,
        answer=answer,
        metadata=meta or {},
        instance_id=0,
    )


# ------------------------------ #
# GSM8KDataset                   #
# ------------------------------ #


def test_dataset():
    rows = [
        {
            "question": "Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
            "answer": "Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
            "metadata": {},
        },
        {
            "question": "Tom has 10 books. He buys 3 more. How many does he have?",
            "answer": "Tom has 10 books.\nHe buys 3 more.\n10 + 3 = 13\n#### 13",
        },
    ]
    ds = GSM8KDataset(rows)
    assert len(ds) == 2
    assert ds.schema is GSM8KInstance
    items = list(ds)
    assert (
        items[0].question
        == "Janet has 5 apples. She gives 2 to Bob. How many does she have left?"
    )
    assert "#### 3" in items[0].answer
    assert items[0].instance_id == 0
    assert items[1].instance_id == 1


def test_dataset_from_hf():
    """Test loading GSM8K from HuggingFace (requires network)."""
    ds = GSM8KDataset.from_hf(
        split="test",
        max_instances=5,
        shuffle=False,
    )
    assert len(ds) == 5
    assert ds.schema is GSM8KInstance
    for instance in ds:
        assert isinstance(instance, GSM8KInstance)
        assert instance.question
        assert instance.answer


# ------------------------------ #
# Answer Extraction              #
# ------------------------------ #


def test_extract_answer_with_marker():
    """Test extracting answer with #### marker."""
    text = "Let me solve this step by step.\n5 + 3 = 8\n#### 8"
    assert extract_answer(text) == 8.0


def test_extract_answer_with_text_marker():
    """Test extracting answer with text markers."""
    text = "The answer is 42."
    assert extract_answer(text) == 42.0

    text = "Answer: 100"
    assert extract_answer(text) == 100.0


def test_extract_answer_last_number():
    """Test extracting the last number when no marker is found."""
    text = "I think the answer might be 5 or maybe 7."
    assert extract_answer(text) == 7.0


def test_extract_answer_no_number():
    """Test when no number is found."""
    text = "I don't know the answer."
    assert extract_answer(text) is None


def test_extract_answer_with_markdown():
    """Test extracting answer when markdown code blocks are present."""
    text = "```python\nx = 5\n```\nThe answer is 10."
    assert extract_answer(text) == 10.0


def test_extract_answer_negative():
    """Test extracting negative numbers."""
    text = "The temperature dropped to -5 degrees."
    assert extract_answer(text) == -5.0


def test_extract_answer_decimal():
    """Test extracting decimal numbers."""
    text = "The result is 3.14."
    assert extract_answer(text) == 3.14


def test_extract_ground_truth():
    """Test extracting ground truth from GSM8K answer format."""
    answer = "Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3"
    assert extract_ground_truth(answer) == 3.0


def test_extract_ground_truth_no_marker():
    """Test extracting ground truth when no marker is present."""
    answer = "The answer is 42."
    assert extract_ground_truth(answer) == 42.0


# ------------------------------ #
# GSM8KEvaluator                 #
# ------------------------------ #


@pytest.fixture
def evaluator():
    return GSM8KEvaluator(tolerance=1e-6)


def test_evaluate_sample_correct(evaluator):
    """Test evaluation with correct answer."""
    instance = GSM8KInstance(
        question="Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
        answer="Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
        instance_id=0,
    )
    response = "Janet has 5 apples and gives 2 away.\n5 - 2 = 3\n#### 3"
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 1.0
    assert result.desc == "correct"
    assert result.metadata["predicted"] == 3.0
    assert result.metadata["ground_truth"] == 3.0


def test_evaluate_sample_incorrect(evaluator):
    """Test evaluation with incorrect answer."""
    instance = GSM8KInstance(
        question="Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
        answer="Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
        instance_id=0,
    )
    response = "The answer is 4."
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 0.0
    assert result.desc == "incorrect"
    assert result.metadata["predicted"] == 4.0
    assert result.metadata["ground_truth"] == 3.0


def test_evaluate_sample_no_answer(evaluator):
    """Test evaluation when no answer is found in response."""
    instance = GSM8KInstance(
        question="Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
        answer="Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
        instance_id=0,
    )
    response = "I'm not sure how to solve this problem."
    result = evaluator.evaluate_sample(instance, response)
    assert result.score == 0.0
    assert result.desc == "no_answer_found"


def test_evaluate_sample_empty_response(evaluator):
    """Test evaluation with empty response."""
    instance = GSM8KInstance(
        question="Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
        answer="Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
        instance_id=0,
    )
    result = evaluator.evaluate_sample(instance, "")
    assert result.score == 0.0
    assert result.desc == "empty_response"


def test_evaluate_sample_tolerance(evaluator):
    """Test that tolerance is respected for floating point comparisons."""
    evaluator.tolerance = 0.1
    instance = GSM8KInstance(
        question="What is 1/3?",
        answer="1/3 = 0.333...\n#### 0.333",
        instance_id=0,
    )
    response = "The answer is 0.33."
    result = evaluator.evaluate_sample(instance, response)
    # 0.33 vs 0.333 with tolerance 0.1 should be correct
    assert result.score == 1.0


def test_evaluate_ensemble(evaluator):
    """Test ensemble evaluation."""
    instance = GSM8KInstance(
        question="Janet has 5 apples. She gives 2 to Bob. How many does she have left?",
        answer="Janet starts with 5 apples.\nShe gives 2 to Bob.\n5 - 2 = 3\n#### 3",
        instance_id=0,
    )

    ensemble_result = evaluator.evaluate_ensemble(
        instance,
        ModelOutput(
            responses=[
                ModelResponse(response="The answer is 3.", weight=0.7),
                ModelResponse(response="The answer is 4.", weight=0.3),
            ],
            runtime_seconds=0.1,
        ),
    )

    assert ensemble_result["weighted_accuracy"] == 0.7
    assert ensemble_result["runtime_seconds"] == 0.1
    assert len(ensemble_result["results"]) == 2


def test_evaluate_sample_with_various_formats(evaluator):
    """Test evaluation with various response formats."""
    instance = GSM8KInstance(
        question="What is 2 + 2?",
        answer="2 + 2 = 4\n#### 4",
        instance_id=0,
    )

    # Test different response formats
    test_cases = [
        ("The answer is 4.", 1.0),
        ("#### 4", 1.0),
        ("Therefore, the answer is 4.", 1.0),
        ("So 2 + 2 equals 4.", 1.0),
        ("The answer is 5.", 0.0),
    ]

    for response, expected_score in test_cases:
        result = evaluator.evaluate_sample(instance, response)
        assert result.score == expected_score, f"Failed for response: {response}"


# ------------------------------ #
# Prompt Formatters              #
# ------------------------------ #


def test_prompt_formatters():
    """Test that all prompt formatters work correctly."""

    # Create a mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            return list(text.encode())

        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            # Simple mock implementation
            return list(" ".join(m.get("content", "") for m in messages).encode())

    tokenizer = MockTokenizer()
    instance = GSM8KInstance(
        question="What is 2 + 2?",
        answer="2 + 2 = 4\n#### 4",
        instance_id=0,
    )

    # Test default formatter
    prompt_ids = default_prompt_formatter(tokenizer, instance, use_chat_format=False)
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0

    # Test chain-of-thought formatter
    prompt_ids = chain_of_thought_prompt_formatter(
        tokenizer, instance, use_chat_format=False
    )
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0

    # Test direct answer formatter
    prompt_ids = direct_answer_prompt_formatter(
        tokenizer, instance, use_chat_format=False
    )
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0

    # Test few-shot formatter
    prompt_ids = few_shot_prompt_formatter(tokenizer, instance, use_chat_format=False)
    assert isinstance(prompt_ids, list)
    assert len(prompt_ids) > 0


def test_few_shot_examples():
    """Test that few-shot examples are properly formatted."""
    assert isinstance(FEW_SHOT_EXAMPLES, list)
    assert len(FEW_SHOT_EXAMPLES) > 0
    for example in FEW_SHOT_EXAMPLES:
        assert isinstance(example, tuple)
        assert len(example) == 2
        assert isinstance(example[0], str)
        assert isinstance(example[1], str)
