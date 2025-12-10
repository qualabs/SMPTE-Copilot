"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .loaders import DocumentLoader, PyMuPDFLoader, LoaderFactory
from .chunkers import Chunker, LangChainChunker, ChunkerFactory
from .embeddings import Embeddings, ChunkEmbedder, EmbeddingModelFactory
from .vector_stores import VectorStore, VectorStoreFactory, ingest_chunks_with_embeddings
from .retrievers import Retriever, DocumentRetriever, RetrieverFactory
from .pipeline import RetrievalPipeline
from .config import Config, get_config

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
    "ChunkEmbedder",
    "DocumentRetriever",
    "RetrievalPipeline",
    # Factories
    "LoaderFactory",
    "ChunkerFactory",
    "EmbeddingModelFactory",
    "VectorStoreFactory",
    "RetrieverFactory",
    # Helpers
    "ingest_chunks_with_embeddings",
    # Config
    "Config",
    "get_config",
]

