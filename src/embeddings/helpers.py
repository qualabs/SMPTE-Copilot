"""Helper class for working with embeddings."""
from __future__ import annotations

from typing import List

from langchain.schema import Document

from .protocol import Embeddings
from .types import EmbeddingModelType


class EmbeddingHelper:
    """Static helper class for embedding operations."""
    
    @staticmethod
    def embed_chunks(
        embedding_model: Embeddings,
        chunks: List[Document],
        model_name: EmbeddingModelType,
    ) -> List[Document]:
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

        # Extract texts for embedding
        texts = [chunk.page_content for chunk in chunks]

        # Generate embeddings
        embeddings = embedding_model.embed_documents(texts)

        # Add embeddings to document metadata
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            # Create a copy to avoid modifying original
            embedded_chunk = Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    "embedding": embedding,
                    "embedding_model": model_name.value,  # Track which model was used
                }
            )
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks

