"""Factory for creating chunker implementations."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .langchain_chunker import create_langchain_chunker
from .protocol import Chunker
from .types import ChunkerType


class ChunkerFactory:
    """Factory for creating chunker implementations. Easily extensible."""

    _registry: ClassVar[dict[ChunkerType, Callable[[dict[str, Any]], Chunker]]] = {}

    @classmethod
    def register(cls, chunker_type: ChunkerType):
        """Register a new chunker factory.

        Parameters
        ----------
        chunker_type
            Type to register the chunker under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Chunker]):
            cls._registry[chunker_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, chunker_type: ChunkerType, **kwargs) -> Chunker:
        """Create a chunker by type.

        Parameters
        ----------
        chunker_type
            Type of the chunker to create.
        **kwargs
            Additional arguments passed to the chunker factory.

        Returns
        -------
        Chunker instance.
        """
        if chunker_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown chunker: {chunker_type}. "
                f"Available chunkers: {available}"
            )
        return cls._registry[chunker_type](kwargs)

ChunkerFactory.register(ChunkerType.LANGCHAIN)(create_langchain_chunker)
