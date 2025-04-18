from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any
from pydantic import BaseModel

from .model import ModelOutput

T = TypeVar("T", bound=BaseModel)


class EvaluationResult(BaseModel):
    """Class for storing evaluation results."""

    score: float
    desc: str
    metadata: Dict[str, Any] = {}


class Evaluator(Generic[T], ABC):
    """Base class for evaluators that handle response evaluation.

    Args:
        T: The Pydantic model type that defines the schema for dataset instances.
    """

    @abstractmethod
    def evaluate_response(self, instance, response):
        """Evaluate a single response for correctness.

        Args:
            instance (T): The dataset instance being evaluated.
            response (str): The model's response text.

        Returns:
            (EvaluationResult): The evaluation result.
        """
        pass  # pragma: no cover

    def evaluate_ensemble(self, instance: T, output: ModelOutput) -> Dict[str, Any]:
        """Evaluate the complete model output, including ensemble responses.

        Default implementation returns posterior weighted accuracy.
        Override this method to add custom evaluation metrics.

        Args:
            instance (T): The dataset instance being evaluated.
            output (ModelOutput): The complete model output including ensemble responses.

        Returns:
            (Dict[str, Any]): Dictionary containing evaluation metrics.
        """
        weighted_accuracy = 0.0
        results = []
        for response in output.responses:
            result = self.evaluate_response(instance, response.text)
            weighted_accuracy += result.score * response.prob
            results.append(
                {
                    "score": result.score,
                    "desc": result.desc,
                    "metadata": result.metadata,
                }
            )

        return {
            "weighted_accuracy": weighted_accuracy,
            "runtime_seconds": output.runtime_seconds,
            "results": results,
        }
