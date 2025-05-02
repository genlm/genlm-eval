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
    """Base class for evaluators that handle response evaluation."""

    @abstractmethod
    def evaluate_sample(self, instance, response):
        """Evaluate a single response for correctness.

        Args:
            instance (T): The dataset instance being evaluated.
            response (Any): The model's response, which is given by the response attribute of a `ModelOutput` object.

        Returns:
            (EvaluationResult): The evaluation result.
        """
        pass  # pragma: no cover

    def evaluate_ensemble(self, instance: T, output: ModelOutput) -> Dict[str, Any]:
        """Evaluate the complete ensemble of weighted samples using weighted accuracy.

        Args:
            instance (T): The dataset instance being evaluated.
            output (ModelOutput): The complete model output including ensemble responses.

        Returns:
            (Dict[str, Any]): Dictionary containing evaluation metrics.
        """
        weighted_accuracy = 0.0
        results = []
        for response in output.responses:
            result = self.evaluate_sample(instance, response.response)
            weighted_accuracy += result.score * response.weight
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
