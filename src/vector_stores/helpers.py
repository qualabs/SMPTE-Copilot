"""Helper class for working with vector stores."""
from __future__ import annotations

from langchain.schema import Document

from ..embeddings.constants import EMBEDDING_METADATA_KEY
from .constants import CHUNK_ID_PREFIX
from .protocol import VectorStore


class VectorStoreHelper:
    """Static helper class for vector store operations."""

    @staticmethod
    def ingest_chunks_with_embeddings(
        vector_store: VectorStore,
        chunks: list[Document],
    ) -> None:
        """Ingest document chunks with embeddings into the vector store.

        This helper method intelligently handles pre-computed embeddings.
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

        has_embeddings = any(EMBEDDING_METADATA_KEY in chunk.metadata for chunk in chunks)

        # Use add_texts with pre-computed embeddings if available
        # This is more efficient than letting the store recompute embeddings
        if has_embeddings:
            texts = [chunk.page_content for chunk in chunks]
            embeddings = [chunk.metadata.get(EMBEDDING_METADATA_KEY) for chunk in chunks]
            metadatas = [
                {k: v for k, v in chunk.metadata.items() if k != EMBEDDING_METADATA_KEY}
                for chunk in chunks
            ]
            ids = [f"{CHUNK_ID_PREFIX}{i}" for i in range(len(chunks))]

            vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            return

        vector_store.add_documents(chunks)

