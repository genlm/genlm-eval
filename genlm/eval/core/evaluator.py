from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Dict, Any
from pydantic import BaseModel

from .model import ModelOutput, ModelResponse

T = TypeVar("T", bound=BaseModel)


class Evaluator(Generic[T], ABC):
    """Base class for evaluators that handle prompt generation and response evaluation.

    Args:
        T: The Pydantic model type that defines the schema for dataset instances.
    """

    @abstractmethod
    def evaluate_response(self, instance: T, response: str) -> bool:
        """Evaluate a single response for correctness.

        Args:
            instance: The dataset instance being evaluated.
            response: The model's response text.

        Returns:
            bool: Whether the response is correct.
        """
        pass

    def evaluate_ensemble(self, instance: T, output: ModelOutput) -> Dict[str, Any]:
        """Evaluate the complete model output, including ensemble responses.

        Default implementation returns posterior weighted accuracy.
        Override this method to add custom evaluation metrics.

        Args:
            instance: The dataset instance being evaluated.
            output: The complete model output including ensemble responses.

        Returns:
            Dict[str, Any]: Dictionary containing evaluation metrics.
        """
        correct_responses: List[ModelResponse] = []
        incorrect_responses: List[ModelResponse] = []

        for response in output.responses:
            if self.evaluate_response(instance, response.text):
                correct_responses.append(response)
            else:
                incorrect_responses.append(response)

        correct_weight = sum(r.weight for r in correct_responses)
        total_weight = sum(r.weight for r in output.responses)
        num_valid = len(output.responses)

        return {
            "weighted_accuracy": correct_weight / total_weight
            if total_weight > 0
            else 0.0,
            "num_correct": len(correct_responses),
            "num_valid": num_valid,
            "runtime_seconds": output.runtime_seconds,
            "num_responses": len(output.responses),
        }
