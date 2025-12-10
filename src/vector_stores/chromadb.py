"""ChromaDB vector store implementation."""
from __future__ import annotations

from typing import Dict, Any

from langchain_community.vectorstores import Chroma
import chromadb  # noqa: F401

from .protocol import VectorStore


def create_chromadb_store(config: Dict[str, Any]) -> VectorStore:
    """Create ChromaDB vector store."""
    from ..embeddings.protocol import Embeddings
    
    # Get configuration
    persist_directory = config.get("persist_directory", "./vector_db")
    collection_name = config.get("collection_name", "rag_documents")
    embedding_function: Embeddings = config.get("embedding_function")
    
    if embedding_function is None:
        raise ValueError(
            "ChromaDB requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )
    
    # Create ChromaDB instance
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

