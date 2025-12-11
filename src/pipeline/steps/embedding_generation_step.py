"""Step that generates embeddings for document chunks."""
from __future__ import annotations

import logging

from langchain.schema import Document

from ...embeddings.constants import EMBEDDING_METADATA_KEY, EMBEDDING_MODEL_METADATA_KEY
from ...embeddings.protocol import Embeddings
from ...embeddings.types import EmbeddingModelType
from ..contexts.ingestion_context import IngestionContext
from ..step import PipelineStep


class EmbeddingGenerationStep:
    """Step that generates embeddings for document chunks."""

    def __init__(self, embedding_model: Embeddings, model_name: EmbeddingModelType):
        """Initialize the embedding generation step.

        Parameters
        ----------
        embedding_model
            Embedding model instance created by EmbeddingModelFactory.
        model_name
            Name of the embedding model (for metadata tracking).
        """
        self.embedding_model = embedding_model
        self.model_name = model_name

    def run(self, context: IngestionContext) -> None:
        """Generate embeddings for all chunks.

        Parameters
        ----------
        context
            Ingestion context with chunks set.
        """
        logger = logging.getLogger()
        if not context.chunks:
            context.mark_failed("No chunks available. Chunk step must run first.")
            return

        logger.info(f"Generating embeddings for {len(context.chunks)} chunks")

        texts = [chunk.page_content for chunk in context.chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        embedded_chunks = []
        for chunk, embedding in zip(context.chunks, embeddings):
            embedded_chunk = Document(
                page_content=chunk.page_content,
                metadata={
                    **chunk.metadata,
                    EMBEDDING_METADATA_KEY: embedding,
                    EMBEDDING_MODEL_METADATA_KEY: self.model_name.value,
                }
            )
            embedded_chunks.append(embedded_chunk)

        context.vectors = embeddings
        context.chunks = embedded_chunks

        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
