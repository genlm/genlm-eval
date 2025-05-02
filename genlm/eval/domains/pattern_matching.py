import regex
import string
import pandas as pd
from genlm.control import Potential
from genlm.eval.util import chat_template_messages
from genlm.eval.core import Dataset, Instance, Evaluator, EvaluationResult


###########
# Dataset #
###########


class PatternMatchingInstance(Instance):
    """Schema for pattern matching instance."""

    pattern: str
    instance_id: int

    def __repr__(self):
        return f"pattern: {self.pattern} (id: {self.instance_id})"


class PatternMatchingDataset(Dataset[PatternMatchingInstance]):
    """Dataset for pattern matching evaluation."""

    def __init__(self, patterns):
        """Initialize the dataset with a list of regex patterns.

        Args:
            patterns (list[str]): List of regex patterns to evaluate.
        """
        self.patterns = patterns

    def __len__(self):
        return len(self.patterns)

    @classmethod
    def from_csv(cls, csv_path, pattern_column):
        """Load patterns from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            pattern_column (str): Name of the column containing regex patterns.

        Returns:
            (PatternMatchingDataset): Dataset initialized with patterns from the CSV.
        """
        patterns = pd.read_csv(csv_path)[pattern_column].to_list()
        return cls(patterns)

    def __iter__(self):
        """Iterate over regex patterns.

        Returns:
            (Iterator[PatternMatchingInstance]): Iterator over regex instances.
        """
        for pattern_id, pattern in enumerate(self.patterns):
            yield PatternMatchingInstance(pattern=pattern, instance_id=pattern_id)

    @property
    def schema(self):
        """Get the schema class for this dataset.

        Returns:
            (type[PatternMatchingInstance]): The Pydantic model class for pattern matching instances.
        """
        return PatternMatchingInstance


#############
# Evaluator #
#############


class PatternMatchingEvaluator(Evaluator[PatternMatchingInstance]):
    """Evaluator for pattern matching."""

    def evaluate_sample(self, instance, response):
        """Evaluate if a response matches the regex pattern.

        Args:
            instance (PatternMatchingInstance): The pattern matching instance being evaluated.
            response (str): The model's response text.

        Returns:
            (EvaluationResult): Evaluation result for whether the response matches the pattern.
        """
        is_valid = regex.compile(instance.pattern).fullmatch(response) is not None
        return EvaluationResult(
            score=int(is_valid), desc="valid" if is_valid else "invalid"
        )


###############
# Model Utils #
###############


class PatternPotential(Potential):
    """Potential function for regex pattern matching."""

    def __init__(self, pattern):
        vocab = list(map(ord, string.printable))
        super().__init__(vocab)
        self.r = regex.compile(pattern)

    async def complete(self, context):
        text = "".join(map(chr, context))
        match = self.r.fullmatch(text) is not None
        return 0.0 if match else float("-inf")

    async def prefix(self, context):
        text = "".join(map(chr, context))
        m = self.r.match(text, partial=True)
        match = m is not None and m.start() == 0 and m.end() == len(text)
        return 0.0 if match else float("-inf")


FEW_SHOT_EXAMPLES = [
    ("(ab)+", "ab"),
    ("(ab|cd)+", "cd"),
    ("[a-z]+", "hello"),
]


SYSTEM_PROMPT = (
    "You are a helpful assistant that generates strings matching regular expressions. "
    + "Only output the exact string that matches the regex pattern, nothing more."
)


def default_prompt_formatter(
    tokenizer,
    instance,
    use_chat_format=False,
    system_prompt=SYSTEM_PROMPT,
    few_shot_examples=FEW_SHOT_EXAMPLES,
):
    """Default prompt formatter for pattern matching.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        instance (PatternMatchingInstance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.
        few_shot_examples (list[tuple[str, str]]): The few shot examples to use. Each example is a tuple of (pattern, response).

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        return tokenizer.apply_chat_template(
            messages=chat_template_messages(
                system_prompt,
                few_shot_examples,
                instance.pattern,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        return tokenizer.encode(
            (
                system_prompt
                + "\n"
                + "\n".join(
                    f"Pattern: {input}\nOutput: {output}"
                    for input, output in few_shot_examples
                )
                + "\n"
                + instance.pattern
            )
        )
