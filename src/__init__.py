"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .loaders import DocumentLoader, PyMuPDFLoader, LoaderFactory, LoaderHelper
from .chunkers import Chunker, LangChainChunker, ChunkerFactory
from .embeddings import Embeddings, EmbeddingModelFactory, EmbeddingHelper
from .vector_stores import VectorStore, VectorStoreFactory, VectorStoreHelper
from .retrievers import Retriever, DocumentRetriever, RetrieverFactory
from .config import Config
from .logger import Logger

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
    "LoaderHelper",
    "VectorStoreHelper",
    "EmbeddingHelper",
    # Config
    "Config",
    # Logger
    "Logger",
]

