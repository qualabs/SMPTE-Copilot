"""Retriever factory for creating retriever implementations."""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .protocol import Retriever
from .similarity_retriever import create_similarity_retriever
from .types import RetrieverType


class RetrieverFactory:
    """Factory for creating retriever implementations. Easily extensible."""

    _registry: ClassVar[dict[RetrieverType, Callable[[dict[str, Any]], Retriever]]] = {}

    @classmethod
    def register(cls, retriever_type: RetrieverType):
        """Register a new retriever factory.

        Parameters
        ----------
        retriever_type
            Type to register the retriever under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], Retriever]):
            cls._registry[retriever_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, retriever_type: RetrieverType, **kwargs) -> Retriever:
        """Create a retriever by type.

        Parameters
        ----------
        retriever_type
            Type of the retriever to create.
        **kwargs
            Additional arguments passed to the retriever factory.

        Returns
        -------
        Retriever instance.
        """
        if retriever_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown retriever: {retriever_type}. "
                f"Available retrievers: {available}"
            )
        return cls._registry[retriever_type](kwargs)

RetrieverFactory.register(RetrieverType.SIMILARITY)(create_similarity_retriever)
