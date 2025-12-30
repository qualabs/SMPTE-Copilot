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
            
            # Inject role-aware access control metadata if provided
            if context.access_tags or context.required_role_strict or context.access_metadata:
                for metadata in metadatas:
                    if context.access_tags:
                        metadata["access_tags"] = context.access_tags
                    if context.required_role_strict:
                        metadata["required_role_strict"] = context.required_role_strict
                    if context.access_metadata:
                        metadata.update(context.access_metadata)
            
            ids = [f"{CHUNK_ID_PREFIX}{i}" for i in range(len(context.chunks))]

            self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            # For documents without embeddings, inject access control metadata
            if context.access_tags or context.required_role_strict or context.access_metadata:
                for chunk in context.chunks:
                    if context.access_tags:
                        chunk.metadata["access_tags"] = context.access_tags
                    if context.required_role_strict:
                        chunk.metadata["required_role_strict"] = context.required_role_strict
                    if context.access_metadata:
                        chunk.metadata.update(context.access_metadata)
            
            self.vector_store.add_documents(context.chunks)

        self.vector_store.persist()
        logger.info(f"Saved {len(context.chunks)} chunks to vector store")
