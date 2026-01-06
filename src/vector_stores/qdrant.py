"""Qdrant vector store implementation."""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..constants import DEFAULT_RETRIEVAL_K
from ..embeddings.protocol import Embeddings
from .constants import DEFAULT_COLLECTION_NAME
from .protocol import VectorStore


class QdrantVectorStoreWrapper:
    """Wrapper for QdrantVectorStore to support pre-computed embeddings in add_texts."""

    def __init__(
        self,
        qdrant_store: QdrantVectorStore,
        client: QdrantClient,
        collection_name: str,
    ):
        """Initialize the wrapper.

        Parameters
        ----------
        qdrant_store
            The underlying QdrantVectorStore instance.
        client
            The QdrantClient instance for direct access.
        collection_name
            Name of the Qdrant collection.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initializing QdrantVectorStoreWrapper for collection: {collection_name}")

        self._store = qdrant_store
        self._client = client
        self._collection_name = collection_name
        self._pending_points: list[PointStruct] = []

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[int]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Add texts to the vector store with pre-computed embeddings.

        Points are accumulated and will be persisted when persist() is called.
        This allows for efficient batching of multiple add_texts calls.

        Parameters
        ----------
        texts
            List of text strings to add.
        metadatas
            Optional list of metadata dictionaries.
        ids
            Optional list of document IDs.
        embeddings
            Pre-computed embedding vectors (always provided in this pipeline).
        """
        if ids is None:
            ids = list(range(len(texts)))
        if metadatas is None:
            metadatas = [{}] * len(texts)

        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            vector = list(embedding)

            payload = {"page_content": text, **(metadata or {})}

            point = PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload,
            )
            self._pending_points.append(point)

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists, raise helpful error if not."""
        try:
            self._client.get_collection(self._collection_name)
        except UnexpectedResponse as e:
            if e.status_code == 404:
                raise ValueError(
                    f"Qdrant collection '{self._collection_name}' does not exist. "
                    f"Please run ingestion first to create the collection and add documents. "
                ) from e
            raise

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[Document]:
        """Search for similar documents."""
        self._ensure_collection_exists()
        if filter is not None:
            return self._store.similarity_search(query, k=k, filter=filter)
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with similarity scores."""
        self._ensure_collection_exists()
        if filter is not None:
            return self._store.similarity_search_with_score(query, k=k, filter=filter)
        return self._store.similarity_search_with_score(query, k=k)

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store."""
        return self._store.add_documents(documents)

    def persist(self) -> None:
        """Persist accumulated points to the Qdrant server.

        This method performs the actual upsert operation with all points
        that were accumulated via add_texts() calls. After persisting,
        the pending points buffer is cleared.
        """
        if not self._pending_points:
            return

        self._logger.info(f"Persisting {len(self._pending_points)} points to Qdrant")

        self._client.upsert(
            collection_name=self._collection_name,
            points=self._pending_points,
        )

        self._pending_points.clear()

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

    logger = logging.getLogger(__name__)

    url = config.get("url", "http://qdrant:6333")
    collection_name = config.get("collection_name", DEFAULT_COLLECTION_NAME)
    embedding_function: Embeddings = config.get("embedding_function")

    if embedding_function is None:
        raise ValueError(
            "Qdrant requires an embedding_function. "
            "Pass it via config: {'embedding_function': embedder.embedding_model}"
        )

    client = QdrantClient(url=url, check_compatibility=False)

    try:
        client.get_collection(collection_name)
        logger.info(f"Qdrant collection already exists: {collection_name}")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            test_embedding = embedding_function.embed_query("test")
            embedding_dim = len(test_embedding)

            logger.info(f"Creating Qdrant collection: {collection_name}")
            logger.info(f"Embedding dimension: {embedding_dim}")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )
        else:
            raise
    except Exception:
        logger.exception("Failed to create Qdrant collection")
        raise

    qdrant_store = QdrantVectorStore(
        client=client,
        embedding=embedding_function,
        collection_name=collection_name,
    )

    return QdrantVectorStoreWrapper(
        qdrant_store=qdrant_store,
        client=client,
        collection_name=collection_name,
    )
