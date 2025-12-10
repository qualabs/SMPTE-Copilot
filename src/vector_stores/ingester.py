"""Vector store ingester implementation.

This module provides VectorStoreIngester, a high-level wrapper around vector stores
that simplifies document ingestion, searching, and persistence operations.

Relationship with VectorStoreFactory:
    - VectorStoreFactory: Creates vector store instances (use this first)
    - VectorStoreIngester: Wrapper that provides convenient methods for working with stores
    
    Example usage:
        # Create store using factory, then pass to ingester
        store = VectorStoreFactory.create(
            "chromadb",
            persist_directory="./db",
            collection_name="docs",
            embedding_function=embedder.embedding_model
        )
        ingester = VectorStoreIngester(vector_store=store)
        ingester.ingest_chunks(chunks)
"""
from __future__ import annotations

from typing import List

from langchain.schema import Document

from .protocol import VectorStore


class VectorStoreIngester:
    """High-level wrapper for ingesting documents into vector stores.
    
    This class provides a convenient interface for working with vector stores.
    It wraps a VectorStore instance (created via VectorStoreFactory) and provides
    methods for ingestion, searching, and persistence.
    
    Examples
    --------
    >>> # Create store using factory first
    >>> store = VectorStoreFactory.create(
    ...     "chromadb",
    ...     persist_directory="./db",
    ...     collection_name="docs",
    ...     embedding_function=embedder.embedding_model
    ... )
    >>> # Then create ingester with the store
    >>> ingester = VectorStoreIngester(vector_store=store)
    >>> ingester.ingest_chunks(chunks)
    """

    def __init__(self, vector_store: VectorStore):
        """Initialize the vector store ingester.

        Parameters
        ----------
        vector_store
            Vector store instance created via VectorStoreFactory.create().
            This must be a pre-initialized vector store object.
        """
        self.vector_store = vector_store

    def ingest_chunks(self, chunks: List[Document]) -> None:
        """Ingest document chunks with embeddings into the vector store.

        Parameters
        ----------
        chunks
            List of Document objects with embeddings in metadata.
            Embeddings should be in chunk.metadata['embedding'].
            If embeddings are present, they will be used; otherwise,
            the vector store will compute them using the embedding function.
        """
        if not chunks:
            return

        # Check if chunks have pre-computed embeddings
        has_embeddings = any("embedding" in chunk.metadata for chunk in chunks)
        
        # Try to use add_texts with pre-computed embeddings if available
        # This is more efficient than letting the store recompute embeddings
        if has_embeddings and hasattr(self.vector_store, "add_texts"):
            try:
                # Extract embeddings and texts separately
                texts = [chunk.page_content for chunk in chunks]
                embeddings = [chunk.metadata.get("embedding") for chunk in chunks]
                metadatas = [
                    {k: v for k, v in chunk.metadata.items() if k != "embedding"}
                    for chunk in chunks
                ]
                ids = [f"chunk_{i}" for i in range(len(chunks))]
                
                # Use add_texts with embeddings (if supported by the store)
                self.vector_store.add_texts(
                    texts=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                return
            except (TypeError, AttributeError):
                # Store doesn't support add_texts with embeddings, fall back
                pass
        
        # Fallback: Let the vector store compute embeddings automatically
        self.vector_store.add_documents(chunks)

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Search the vector store for similar documents.

        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.

        Returns
        -------
        List of Document objects, most similar first.
        """
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """Search the vector store with similarity scores.

        Parameters
        ----------
        query
            Search query text.
        k
            Number of results to return.

        Returns
        -------
        List of tuples: (Document, score), most similar first.
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def persist(self) -> None:
        """Persist the vector store to disk (if supported)."""
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
        elif hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "persist"):
            # For ChromaDB
            self.vector_store._collection.persist()

    def delete_collection(self) -> None:
        """Delete the collection (if supported)."""
        if hasattr(self.vector_store, "delete"):
            self.vector_store.delete()
        elif hasattr(self.vector_store, "_collection"):
            # For ChromaDB
            try:
                self.vector_store._collection.delete()
            except Exception:
                pass

