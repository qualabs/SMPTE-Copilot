"""Factory for creating vector store implementations.

This module provides VectorStoreFactory, which creates instances of different
vector store implementations (ChromaDB, Pinecone, etc.) based on a name.

Usage:
    >>> # Create store using factory
    >>> store = VectorStoreFactory.create(
    ...     "chromadb",
    ...     persist_directory="./db",
    ...     collection_name="docs",
    ...     embedding_function=embedder.embedding_model
    ... )
    >>> # Use store directly or with helper functions
    >>> from src.vector_stores.helpers import ingest_chunks_with_embeddings
    >>> ingest_chunks_with_embeddings(store, chunks)
"""
from __future__ import annotations

from typing import List, Dict, Any, Callable

from .protocol import VectorStore

from .chromadb import create_chromadb_store


class VectorStoreFactory:
    """Factory for creating vector store implementations. Easily extensible."""
    _registry: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Register a new vector store factory.
        
        Parameters
        ----------
        name
            Name to register the vector store under.
        """
        def decorator(factory_func: Callable[[Dict[str, Any]], Any]):
            cls._registry[name] = factory_func
            return factory_func
        return decorator
    
    @classmethod
    def create(cls, store_name: str, **kwargs) -> VectorStore:
        """Create a vector store by name.
        
        Parameters
        ----------
        store_name
            Name of the vector store to create.
        **kwargs
            Additional arguments passed to the store factory.
            
        Returns
        -------
        Vector store instance.
        """
        if store_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown vector store: {store_name}. "
                f"Available stores: {available}"
            )
        return cls._registry[store_name](kwargs)

# Register vector store implementations
VectorStoreFactory.register("chromadb")(create_chromadb_store)
