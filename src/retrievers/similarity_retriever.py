from __future__ import annotations

"""Similarity search retriever implementation."""

from typing import Any

from langchain.schema import Document

from ..constants import DEFAULT_RETRIEVAL_K
from ..vector_stores.protocol import VectorStore
from .protocol import Retriever


class DocumentRetriever:
    """Retrieve relevant documents from vector store using similarity search.

    This is a concrete implementation of the Retriever protocol using
    similarity search on a vector store.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = DEFAULT_RETRIEVAL_K,
        metadata_filter: Any = None,
    ):
        """Initialize the retriever.

        Parameters
        ----------
        vector_store
            Vector store instance (created via VectorStoreFactory).
        k
            Number of documents to retrieve. Default: DEFAULT_RETRIEVAL_K
        metadata_filter
            Optional filter for metadata-based retrieval (e.g., Qdrant Filter).
        """
        self.vector_store = vector_store
        self.k = k
        self.metadata_filter = metadata_filter

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for a query."""
        # Check if vector store supports filtering
        if self.metadata_filter is not None and hasattr(
            self.vector_store, "similarity_search"
        ):
            try:
                # Try to pass filter parameter (Qdrant supports this)
                return self.vector_store.similarity_search(
                    query, k=self.k, filter=self.metadata_filter
                )
            except TypeError:
                # Fallback if vector store doesn't support filter parameter
                return self.vector_store.similarity_search(query, k=self.k)
        return self.vector_store.similarity_search(query, k=self.k)

    def retrieve_with_scores(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents with similarity scores."""
        # Check if vector store supports filtering
        if self.metadata_filter is not None and hasattr(
            self.vector_store, "similarity_search_with_score"
        ):
            try:
                # Try to pass filter parameter (Qdrant supports this)
                return self.vector_store.similarity_search_with_score(
                    query, k=self.k, filter=self.metadata_filter
                )
            except TypeError:
                # Fallback if vector store doesn't support filter parameter
                return self.vector_store.similarity_search_with_score(query, k=self.k)
        return self.vector_store.similarity_search_with_score(query, k=self.k)


def create_similarity_retriever(config: dict[str, Any]) -> Retriever:
    """Create a similarity retriever from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - vector_store: VectorStore (required) - Vector store instance
        - k: int (optional) - Number of documents to retrieve
        - metadata_filter: Any (optional) - Filter for metadata-based retrieval

    Returns
    -------
    Retriever instance.
    """
    vector_store = config.get("vector_store")
    if vector_store is None:
        raise ValueError("vector_store is required for similarity retriever")

    k = config.get("k", DEFAULT_RETRIEVAL_K)

    if k is not None and (not isinstance(k, int) or k <= 0):
        raise ValueError(f"k must be a positive integer, got: {k}")

    metadata_filter = config.get("metadata_filter")

    return DocumentRetriever(vector_store, k=k, metadata_filter=metadata_filter)

