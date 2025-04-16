from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic, Union
from pydantic import BaseModel


class Instance(BaseModel):
    """Base class for dataset instances that conform to a Pydantic schema."""

    instance_id: Union[int, str]


T = TypeVar("T", bound=Instance)


class Dataset(Generic[T], ABC):
    """Base class for datasets that yield instances conforming to a Pydantic schema.

    Args:
        T: The Pydantic model type that defines the schema for dataset instances.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over dataset instances.

        Returns:
            Iterator[T]: An iterator over instances conforming to schema T.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def schema(self) -> type[T]:
        """Get the Pydantic schema class for this dataset.

        Returns:
            type[T]: The Pydantic model class defining the schema.
        """
        pass  # pragma: no cover
