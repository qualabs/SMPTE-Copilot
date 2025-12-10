"""Factory for creating vector store implementations.

This module provides VectorStoreFactory, which creates instances of different
vector store implementations (ChromaDB, Pinecone, etc.) based on a type.

Usage:
    >>> # Create store using factory
    >>> from src.vector_stores.types import VectorStoreType
    >>> store = VectorStoreFactory.create(
    ...     VectorStoreType.CHROMADB,
    ...     persist_directory="./db",
    ...     collection_name="docs",
    ...     embedding_function=embedder.embedding_model
    ... )
    >>> # Use store directly or with helper class
    >>> from src.vector_stores.helpers import VectorStoreHelper
    >>> VectorStoreHelper.ingest_chunks_with_embeddings(store, chunks)
"""
from __future__ import annotations

from typing import Any, Callable, ClassVar

from .chromadb import create_chromadb_store
from .protocol import VectorStore
from .types import VectorStoreType


class VectorStoreFactory:
    """Factory for creating vector store implementations. Easily extensible."""
    _registry: ClassVar[dict[VectorStoreType, Callable[[dict[str, Any]], VectorStore]]] = {}

    @classmethod
    def register(cls, store_type: VectorStoreType):
        """Register a new vector store factory.

        Parameters
        ----------
        store_type
            Type to register the vector store under.
        """
        def decorator(factory_func: Callable[[dict[str, Any]], VectorStore]):
            cls._registry[store_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, store_type: VectorStoreType, **kwargs) -> VectorStore:
        """Create a vector store by type.

        Parameters
        ----------
        store_type
            Type of the vector store to create.
        **kwargs
            Additional arguments passed to the store factory.

        Returns
        -------
        Vector store instance.
        """
        if store_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown vector store: {store_type}. "
                f"Available stores: {available}"
            )
        return cls._registry[store_type](kwargs)

# Register vector store implementations
VectorStoreFactory.register(VectorStoreType.CHROMADB)(create_chromadb_store)
