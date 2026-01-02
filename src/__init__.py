"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .chunkers import ChunkerFactory
from .config import Config
from .embeddings import EmbeddingModelFactory, Embeddings
from .llms import LLMFactory
from .loaders import LoaderFactory, LoaderHelper
from .logger import Logger
from .preprocessing import PreprocessorFactory
from .retrievers import RetrieverFactory
from .vector_stores import VectorStoreFactory, VectorStore

__all__ = [
    # Protocols (used for type hints)
    "Embeddings",
    "VectorStore",
    # Factories
    "LoaderFactory",
    "ChunkerFactory",
    "EmbeddingModelFactory",
    "VectorStoreFactory",
    "RetrieverFactory",
    "LLMFactory",
    "PreprocessorFactory",
    # Helpers
    "LoaderHelper",
    # Config
    "Config",
    # Logger
    "Logger",
]

