"""Protocol definitions for RAG components.

This package defines structural type contracts using typing.Protocol.
These protocols enable type checking and ensure components implement
the required methods without requiring explicit inheritance.
"""

from .vector_store import VectorStore
from .embeddings import Embeddings

__all__ = ["VectorStore", "Embeddings"]

