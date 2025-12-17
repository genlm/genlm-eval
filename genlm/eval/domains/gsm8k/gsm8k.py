import logging
import random
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

from datasets import load_dataset

from genlm.eval.core import EvaluationResult, Instance, Dataset, Evaluator

log = logging.getLogger(__name__)

################
# GSM8K Data   #
################


class GSM8KInstance(Instance):
    """Schema for a GSM8K instance."""

    question: str
    answer: str
    metadata: Dict[str, Any] = {}


class GSM8KDataset(Dataset[GSM8KInstance]):
    """Dataset for GSM8K evaluation (Cobbe et al., 2021)."""

    def __init__(self, rows: List[Mapping[str, Any]]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[GSM8KInstance]:
        for i, row in enumerate(self._rows):
            yield GSM8KInstance(
                question=str(row.get("question", "")).strip(),
                answer=str(row.get("answer", "")).strip(),
                metadata=(row.get("metadata") or {}),
                instance_id=i,
            )

    @property
    def schema(self) -> Type[GSM8KInstance]:
        return GSM8KInstance

    @classmethod
    def from_hf(
        cls,
        split: str = "test",
        max_instances: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 1234,
        cache_dir: Optional[str] = None,
    ) -> "GSM8KDataset":
        """Load and (optionally) filter GSM8K from Hugging Face.

        Args:
            split: Dataset split to load ("train" or "test").
            max_instances: Maximum number of instances to load.
            shuffle: Whether to shuffle the dataset.
            seed: Random seed for shuffling.
            cache_dir: Directory to cache the dataset.

        Returns:
            GSM8KDataset with loaded instances.
        """
        ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
        rows: List[Mapping[str, Any]] = list(ds)

        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(rows)

        if isinstance(max_instances, int) and max_instances >= 0:
            rows = rows[:max_instances]

        log.info("Loaded GSM8K: %d instances (split=%s)", len(rows), split)
        return cls(rows)


################
#   Evaluator  #
################


def extract_answer(text: str) -> Optional[float]:
    """Extract the final numerical answer from a text response.

    This function looks for the last number in the text that appears after
    common answer markers like "####", "The answer is", etc.

    Args:
        text: The text response to extract the answer from.

    Returns:
        The extracted numerical answer, or None if no valid answer is found.
    """
    # Remove markdown code blocks if present
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Try to find answer after common markers (in order of preference)
    patterns = [
        r"####\s*([+-]?\d+(?:\.\d+)?)",  # GSM8K standard format
        r"(?:The answer is|Answer:|answer is|answer:)\s*([+-]?\d+(?:\.\d+)?)",
        r"(?:Therefore|So|Thus|Hence)[^.]*?(?:is|equals?|=\s*)([+-]?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except ValueError:
                continue

    # If no marker found, try to extract numbers but prefer those at the end
    # Look for numbers that appear in the last third of the text
    numbers = re.finditer(r"([+-]?\d+(?:\.\d+)?)", text)
    number_list = [(m.group(1), m.start()) for m in numbers]

    if number_list:
        # If we have numbers, prefer those in the last third of the text
        text_len = len(text)
        threshold = text_len * 2 / 3

        # Filter numbers in the last third
        late_numbers = [(num, pos) for num, pos in number_list if pos >= threshold]

        if late_numbers:
            # Use the last number in the last third
            try:
                return float(late_numbers[-1][0])
            except ValueError:
                pass

        # Fallback to the absolute last number
        try:
            return float(number_list[-1][0])
        except ValueError:
            pass

    return None


def extract_ground_truth(answer: str) -> Optional[float]:
    """Extract the ground truth answer from the GSM8K answer field.

    The answer field typically contains a solution followed by "#### <number>".

    Args:
        answer: The answer field from the dataset.

    Returns:
        The ground truth numerical answer, or None if not found.
    """
    # GSM8K format: solution text followed by "#### <number>"
    match = re.search(r"####\s*([+-]?\d+(?:\.\d+)?)", answer)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Fallback: try to find the last number
    numbers = re.findall(r"([+-]?\d+(?:\.\d+)?)", answer)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


class GSM8KEvaluator(Evaluator[GSM8KInstance]):
    """Evaluator for GSM8K math word problems."""

    def __init__(self, tolerance: float = 1e-6) -> None:
        """Initialize the evaluator.

        Args:
            tolerance: Numerical tolerance for comparing answers.
        """
        self.tolerance = float(tolerance)

    def evaluate_sample(
        self, instance: GSM8KInstance, response: str
    ) -> EvaluationResult:
        """Evaluate a single response for correctness.

        Args:
            instance: The GSM8K instance being evaluated.
            response: The model's response text.

        Returns:
            EvaluationResult with score 1.0 if correct, else 0.0.
        """
        if not response or not response.strip():
            return EvaluationResult(
                score=0.0, desc="empty_response", metadata=instance.metadata
            )

        # Extract ground truth answer
        ground_truth = extract_ground_truth(instance.answer)
        if ground_truth is None:
            return EvaluationResult(
                score=0.0, desc="invalid_ground_truth", metadata=instance.metadata
            )

        # Extract predicted answer
        predicted = extract_answer(response)
        if predicted is None:
            return EvaluationResult(
                score=0.0,
                desc="no_answer_found",
                metadata={**instance.metadata, "response": response[:200]},
            )

        # Compare answers with tolerance
        is_correct = abs(predicted - ground_truth) <= self.tolerance

        return EvaluationResult(
            score=1.0 if is_correct else 0.0,
            desc="correct" if is_correct else "incorrect",
            metadata={
                **instance.metadata,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "difference": abs(predicted - ground_truth),
            },
        )


##########################
# Prompt formatter (LM)  #
##########################

# Few-shot examples for GSM8K
FEW_SHOT_EXAMPLES = [
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
    ),
    (
        "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10",
    ),
]


def default_prompt_formatter(
    tokenizer, instance: GSM8KInstance, use_chat_format: bool = False
) -> List[int]:
    """Default prompt formatter for GSM8K (simple question-only format).

    Args:
        tokenizer: The tokenizer to use.
        instance: The GSM8K instance to format.
        use_chat_format: Whether to use chat format.

    Returns:
        List of token IDs for the prompt.
    """
    prompt = instance.question
    if use_chat_format:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves grade school math word problems step by step.",
            },
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        return tokenizer.encode(prompt)


def chain_of_thought_prompt_formatter(
    tokenizer, instance: GSM8KInstance, use_chat_format: bool = False
) -> List[int]:
    """Chain-of-thought prompt formatter that encourages step-by-step reasoning.

    This formatter explicitly asks the model to show its work and think step by step.

    Args:
        tokenizer: The tokenizer to use.
        instance: The GSM8K instance to format.
        use_chat_format: Whether to use chat format.

    Returns:
        List of token IDs for the prompt.
    """
    if use_chat_format:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves grade school math word problems. Show your work step by step, then provide the final answer.",
            },
            {
                "role": "user",
                "content": f"Solve this problem step by step:\n\n{instance.question}\n\nShow your reasoning and then provide the final answer.",
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt = f"Solve this problem step by step:\n\n{instance.question}\n\nShow your reasoning and then provide the final answer."
        return tokenizer.encode(prompt)


def direct_answer_prompt_formatter(
    tokenizer, instance: GSM8KInstance, use_chat_format: bool = False
) -> List[int]:
    """Direct answer prompt formatter that asks for just the numerical answer.

    This formatter asks the model to provide only the final answer without showing work.

    Args:
        tokenizer: The tokenizer to use.
        instance: The GSM8K instance to format.
        use_chat_format: Whether to use chat format.

    Returns:
        List of token IDs for the prompt.
    """
    if use_chat_format:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves grade school math word problems. Provide only the numerical answer.",
            },
            {
                "role": "user",
                "content": f"{instance.question}\n\nProvide only the numerical answer.",
            },
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt = f"{instance.question}\n\nProvide only the numerical answer."
        return tokenizer.encode(prompt)


def few_shot_prompt_formatter(
    tokenizer,
    instance: GSM8KInstance,
    use_chat_format: bool = False,
    few_shot_examples: Optional[List[tuple[str, str]]] = None,
) -> List[int]:
    """Few-shot prompt formatter that includes example problems and solutions.

    This formatter includes example problems with their solutions to guide the model.

    Args:
        tokenizer: The tokenizer to use.
        instance: The GSM8K instance to format.
        use_chat_format: Whether to use chat format.
        few_shot_examples: Optional list of (question, answer) tuples. Defaults to FEW_SHOT_EXAMPLES.

    Returns:
        List of token IDs for the prompt.
    """
    if few_shot_examples is None:
        few_shot_examples = FEW_SHOT_EXAMPLES

    if use_chat_format:
        from genlm.eval.util import chat_template_messages

        system_prompt = "You are a helpful assistant that solves grade school math word problems step by step."
        messages = chat_template_messages(
            system_prompt,
            few_shot_examples,
            instance.question,
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt_parts = [
            "Solve the following math word problems step by step. Show your work and provide the final answer in the format: #### <number>\n\n"
        ]
        for q, a in few_shot_examples:
            prompt_parts.append(f"Question: {q}\n")
            prompt_parts.append(f"Answer: {a}\n\n")
        prompt_parts.append(f"Question: {instance.question}\n")
        prompt_parts.append("Answer:")
        prompt = "".join(prompt_parts)
        return tokenizer.encode(prompt)
