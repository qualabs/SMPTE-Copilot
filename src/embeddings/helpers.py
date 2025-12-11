"""Helper class for working with embeddings."""
from __future__ import annotations

from langchain.schema import Document

from .constants import EMBEDDING_METADATA_KEY, EMBEDDING_MODEL_METADATA_KEY
from .protocol import Embeddings
from .types import EmbeddingModelType


class EmbeddingHelper:
    """Static helper class for embedding operations."""

    @staticmethod
    def embed_chunks(
        embedding_model: Embeddings,
        chunks: list[Document],
        model_name: EmbeddingModelType,
    ) -> list[Document]:
        """Embed a list of document chunks and add embeddings to metadata.

        Parameters
        ----------
        embedding_model
            Embedding model instance (created via EmbeddingModelFactory).
        chunks
            List of LangChain Document objects to embed.
        model_name
            Type of the model used (for tracking in metadata).

        Returns
        -------
        List of Document objects with embeddings stored in metadata.
        """
        if not chunks:
            return []

        texts = [chunk.page_content for chunk in chunks]

        embeddings = embedding_model.embed_documents(texts)

        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    EMBEDDING_METADATA_KEY: embedding,
                    EMBEDDING_MODEL_METADATA_KEY: model_name.value,  # Track which model was used
                }
            )
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

