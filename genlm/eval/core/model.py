from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ModelResponse(BaseModel):
    """Container for a single response from the model."""

    text: str
    prob: float
    metadata: Optional[Dict[str, Any]] = None


class ModelOutput(BaseModel):
    """Container for the complete model output, including ensemble and runtime info."""

    responses: List[ModelResponse]
    runtime_seconds: float
    metadata: Optional[Dict[str, Any]] = None


class ModelAdaptor(ABC):
    """Base class for model adaptors that handle interaction with language models."""

    @abstractmethod
    async def generate(self, instance) -> ModelOutput:
        """Generate responses for a given prompt.

        Args:
            instance: The input instance to send to the model.

        Returns:
            ModelOutput: Container with weighted ensemble responses and metadata.
        """
        pass  # pragma: no cover

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about the model."""
        pass  # pragma: no cover
