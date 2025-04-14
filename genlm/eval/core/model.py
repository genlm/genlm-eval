from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ModelResponse:
    """Container for a single response from the model."""

    text: str
    prob: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "text": self.text,
            "prob": self.prob,
            "metadata": self.metadata,
        }

    def __repr__(self):
        return f"ModelResponse(\n\ttext={self.text},\n\tprob={self.prob},\n\tmetadata={self.metadata}\n)"


@dataclass
class ModelOutput:
    """Container for the complete model output, including ensemble and runtime info."""

    responses: List[ModelResponse]
    runtime_seconds: float
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "responses": [r.to_dict() for r in self.responses],
            "runtime_seconds": self.runtime_seconds,
            "metadata": self.metadata,
        }

    def __repr__(self):
        return f"ModelOutput(\n\truntime_seconds={self.runtime_seconds},\n\tresponses={self.responses},\n\tmetadata={self.metadata}\n)"


class ModelAdaptor(ABC):
    """Base class for model adaptors that handle interaction with language models."""

    @abstractmethod
    async def generate(self, instance) -> ModelOutput:
        """Generate responses for a given prompt.

        Args:
            instance: The input instance to send to the model.

        Returns:
            ModelOutput: Container with weighted responses and metadata.
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the model."""
        pass
