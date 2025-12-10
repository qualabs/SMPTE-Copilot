"""Helper functions for working with vector stores."""
from __future__ import annotations

from typing import List

from langchain.schema import Document

from .protocol import VectorStore


def ingest_chunks_with_embeddings(
    vector_store: VectorStore,
    chunks: List[Document],
) -> None:
    """Ingest document chunks with embeddings into the vector store.
    
    This helper function intelligently handles pre-computed embeddings.
    If embeddings are present in chunk metadata, it uses them directly
    (more efficient). Otherwise, it lets the vector store compute them.
    
    Parameters
    ----------
    vector_store
        Vector store instance to ingest into.
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
    if has_embeddings and hasattr(vector_store, "add_texts"):
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
            vector_store.add_texts(
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
    vector_store.add_documents(chunks)

