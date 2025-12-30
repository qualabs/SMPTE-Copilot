"""Qdrant vector store implementation."""
from __future__ import annotations

from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ..embeddings.protocol import Embeddings
from .constants import DEFAULT_COLLECTION_NAME
from .protocol import VectorStore


def create_qdrant_store(config: dict[str, Any]) -> VectorStore:
    """Create a Qdrant vector store from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - embedding_function: Embeddings (required) - Embedding model instance
        - url: str (optional) - URL to Qdrant server (default: http://localhost:6333)
        - collection_name: str (optional) - Name of the collection

    Returns
    -------
    VectorStore instance.

    Raises
    ------
    ValueError
        If embedding_function is not provided.
    ImportError
        If required packages are not installed.
    """
    try:
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "Qdrant requires 'qdrant-client' and 'langchain-qdrant' packages. "
            "Install with: pip install qdrant-client langchain-qdrant"
        ) from exc

    url = config.get("url", "http://localhost:6333")
    collection_name = config.get("collection_name", DEFAULT_COLLECTION_NAME)
    embedding_function = config.get("embedding_function")

    if embedding_function is None:
        raise ValueError(
            "Qdrant requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )

    # Create Qdrant client
    print("QDRANT URL: ", url)
    client = QdrantClient(url=url)

    # Check if collection exists, if not create it manually
    if not client.collection_exists(collection_name):
        # Get embedding dimension from the embedding function
        # Create a test embedding to determine the dimension
        test_embedding = embedding_function.embed_query("test")
        embedding_dim = len(test_embedding)

        # Create collection with proper vector configuration
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    # Create Qdrant vector store using the new langchain-qdrant package
    # Pass the client directly for proper initialization
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_function,
    )
