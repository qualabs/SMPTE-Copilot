from __future__ import annotations

"""Protocol for vector store implementations."""

from typing import Any, Optional, Protocol

from langchain.schema import Document

from ..constants import DEFAULT_RETRIEVAL_K


class VectorStore(Protocol):
    """Protocol for vector store implementations.

    Any class implementing these methods is compatible with VectorStore,
    regardless of inheritance hierarchy. This allows LangChain's vector
    stores (Chroma, Pinecone, etc.) to work seamlessly without modification.

    """

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[Document]:
        """Search for similar documents.

        Parameters
        ----------
        query
            Query string to search for.
        k
            Number of documents to retrieve.
        filter
            Optional metadata filter (vector store specific).
        """
        ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        filter: Optional[Any] = None,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with similarity scores.

        Parameters
        ----------
        query
            Query string to search for.
        k
            Number of documents to retrieve.
        filter
            Optional metadata filter (vector store specific).
        """
        ...

    def add_documents(
        self,
        documents: list[Document]
    ) -> list[str]:
        """Add documents to the vector store."""
        ...

    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[int]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Add texts to the vector store.

        Add texts directly with optional embeddings and metadata.
        This method is recommended for better performance when embeddings
        are pre-computed.

        Parameters
        ----------
        texts
            List of text strings to add.
        metadatas
            Optional list of metadata dictionaries.
        ids
            Optional list of document IDs.
        embeddings
            Optional list of pre-computed embedding vectors.

        """
        ...

    def persist(self) -> None:
        """Persist the vector store to disk."""
        ...

