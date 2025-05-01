from pydantic import BaseModel
from typing import List, Dict, Any, Protocol, runtime_checkable, Optional
from .dataset import Instance


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


@runtime_checkable
class ModelAdaptor(Protocol):
    """Protocol for model adapters. Must be async callable that takes a dataset Instance and returns a ModelOutput."""

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        """Process an instance and return a ModelOutput."""
        ...
