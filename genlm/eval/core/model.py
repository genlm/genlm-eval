from pydantic import BaseModel
from typing import List, Dict, Any, Protocol, runtime_checkable, Optional
from .dataset import Instance


class ModelResponse(BaseModel):
    """Single model response containing generated text, probability, and optional metadata."""

    text: str
    prob: float
    metadata: Optional[Dict[str, Any]] = None


class ModelOutput(BaseModel):
    """Collection of model responses with execution metadata."""

    responses: List[ModelResponse]
    runtime_seconds: float
    metadata: Optional[Dict[str, Any]] = None


@runtime_checkable
class ModelAdaptor(Protocol):
    """Protocol for async model adapters that process instances into model outputs."""

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        """Process an instance and generate model outputs.

        Args:
            instance (Instance): Input dataset instance to process
            output_dir (str): Directory for saving any intermediate results
            replicate (int): Replicate index for multiple evaluation runs

        Returns:
            (ModelOutput): Model output containing responses and runtime information
        """
        ...  # pragma: no cover
