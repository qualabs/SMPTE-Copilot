"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .chunkers import Chunker, ChunkerFactory, LangChainChunker
from .config import Config
from .embeddings import EmbeddingHelper, EmbeddingModelFactory, Embeddings
from .loaders import DocumentLoader, LoaderFactory, LoaderHelper, PyMuPDFLoader
from .logger import Logger
from .retrievers import DocumentRetriever, Retriever, RetrieverFactory
from .vector_stores import VectorStore, VectorStoreFactory, VectorStoreHelper

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

