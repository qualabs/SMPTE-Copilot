"""
RAG ingestion helpers.

Utilities for converting PDF documents into Markdown, chunking, embedding,
vector storage, and retrieval.
"""

from .loaders import DocumentLoader, PyMuPDFLoader
from .chunkers import Chunker, LangChainChunker
from .embeddings import Embeddings, ChunkEmbedder
from .vector_stores import VectorStore, VectorStoreIngester
from .retrievers import Retriever, DocumentRetriever
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
    "VectorStoreIngester",
    "DocumentRetriever",
    "RetrievalPipeline",
    # Config
    "Config",
    "get_config",
]

