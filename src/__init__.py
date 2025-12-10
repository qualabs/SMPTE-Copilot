"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .loaders import DocumentLoader, PyMuPDFLoader, LoaderFactory
from .chunkers import Chunker, LangChainChunker, ChunkerFactory
from .embeddings import Embeddings, EmbeddingModelFactory, EmbeddingHelper
from .vector_stores import VectorStore, VectorStoreFactory, VectorStoreHelper
from .retrievers import Retriever, DocumentRetriever, RetrieverFactory
from .config import Config

__all__ = [
    # Protocols
    "DocumentLoader",
    "Chunker",
    "Embeddings",
    "VectorStore",
    "Retriever",
    # Implementations
    "PyMuPDFLoader",
    "LangChainChunker",
    "DocumentRetriever",
    # Factories
    "LoaderFactory",
    "ChunkerFactory",
    "EmbeddingModelFactory",
    "VectorStoreFactory",
    "RetrieverFactory",
    # Helpers
    "VectorStoreHelper",
    "EmbeddingHelper",
    # Config
    "Config",
]

