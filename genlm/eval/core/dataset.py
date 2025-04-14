from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


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
        pass

    @property
    @abstractmethod
    def schema(self) -> type[T]:
        """Get the Pydantic schema class for this dataset.

        Returns:
            type[T]: The Pydantic model class defining the schema.
        """
        pass
