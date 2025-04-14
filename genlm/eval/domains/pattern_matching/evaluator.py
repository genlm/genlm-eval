import regex
from genlm.eval.core import Evaluator
from .dataset import PatternMatchingInstance


class PatternMatchingEvaluator(Evaluator[PatternMatchingInstance]):
    """Evaluator for pattern matching."""

    def evaluate_response(self, instance, response):
        """Evaluate if a response matches the regex pattern.

        Args:
            instance (PatternMatchingInstance): The pattern matching instance being evaluated.
            response (str): The model's response text.

        Returns:
            (bool): Whether the response matches the pattern.
        """
        try:
            return regex.compile(instance.pattern).fullmatch(response) is not None
        except (regex.error, TypeError):
            return False
