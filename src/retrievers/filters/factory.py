"""Factory for creating filter builders."""
from __future__ import annotations

from typing import ClassVar, Callable

from ...vector_stores.types import VectorStoreType
from .qdrant import QdrantFilterBuilder
from .protocol import FilterBuilder


class FilterBuilderFactory:
    """Factory for creating filter builders based on vector store type."""

    _registry: ClassVar[dict[VectorStoreType, Callable[[], FilterBuilder]]] = {}

    @classmethod
    def register(cls, store_type: VectorStoreType):
        """Register a filter builder for a vector store type.

        Parameters
        ----------
        store_type
            Vector store type to register the builder for.
        """
        def decorator(builder_factory: Callable[[], FilterBuilder]):
            cls._registry[store_type] = builder_factory
            return builder_factory
        return decorator

    @classmethod
    def create(cls, store_type: VectorStoreType) -> FilterBuilder:
        """Create a filter builder for the specified vector store type.

        Parameters
        ----------
        store_type
            Type of the vector store.

        Returns
        -------
        FilterBuilder
            Filter builder instance for the vector store.

        Raises
        ------
        ValueError
            If no builder is registered for the vector store type.
        """
        if store_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"No filter builder registered for vector store: {store_type}. "
                f"Available builders: {available}"
            )
        return cls._registry[store_type]()


FilterBuilderFactory.register(VectorStoreType.QDRANT)(QdrantFilterBuilder)

