"""Step that saves chunks with embeddings to the vector store."""
from __future__ import annotations

import logging

from ...embeddings.constants import EMBEDDING_METADATA_KEY
from ...vector_stores.constants import CHUNK_ID_PREFIX
from ...vector_stores.protocol import VectorStore
from ..contexts.ingestion_context import IngestionContext
from ..step import PipelineStep


class SaveStep:
    """Step that saves chunks with embeddings to the vector store."""

    def __init__(self, vector_store: VectorStore):
        """Initialize the save step.

        Parameters
        ----------
        vector_store
            Vector store instance created by VectorStoreFactory.
        """
        self.vector_store = vector_store

    def run(self, context: IngestionContext) -> None:
        """Save chunks with embeddings to the vector store.

        Parameters
        ----------
        context
            Ingestion context with chunks and vectors set.
        """
        logger = logging.getLogger()
        if not context.chunks:
            context.mark_failed("No chunks available. Embedding step must run first.")
            return

        logger.info(f"Saving {len(context.chunks)} chunks to vector store")

        has_embeddings = any(
            EMBEDDING_METADATA_KEY in chunk.metadata for chunk in context.chunks
        )

        if has_embeddings:
            texts = [chunk.page_content for chunk in context.chunks]
            embeddings = [
                chunk.metadata.get(EMBEDDING_METADATA_KEY) for chunk in context.chunks
            ]
            metadatas = [
                {k: v for k, v in chunk.metadata.items() if k != EMBEDDING_METADATA_KEY}
                for chunk in context.chunks
            ]
            ids = [f"{CHUNK_ID_PREFIX}{i}" for i in range(len(context.chunks))]

            self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            self.vector_store.add_documents(context.chunks)

        self.vector_store.persist()
        logger.info(f"Saved {len(context.chunks)} chunks to vector store")
