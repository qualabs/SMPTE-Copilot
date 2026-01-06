from __future__ import annotations

"""Step that saves chunks with embeddings to the vector store."""

import hashlib
import logging

from ...embeddings.constants import EMBEDDING_METADATA_KEY
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

    def _inject_access_metadata(self, metadata: dict, context: IngestionContext) -> None:
        """Inject access control metadata into a metadata dictionary.
        
        Prioritizes file metadata over global config. If metadata already
        contains access_tags, those are preserved. Otherwise, falls back to context values.

        Parameters
        ----------
        metadata
            Metadata dictionary to update (modified in place).
        context
            Ingestion context with access control settings.
        """
        if "access_tags" not in metadata and context.access_tags:
            metadata["access_tags"] = context.access_tags
        
        if context.access_metadata:
            metadata.update(context.access_metadata)

    def run(self, context: IngestionContext) -> None:
        """Save chunks with embeddings to the vector store.

        Parameters
        ----------
        context
            Ingestion context with chunks and vectors set.
        """
        logger = logging.getLogger(__name__)
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
            
            for metadata in metadatas:
                self._inject_access_metadata(metadata, context)
            
            file_path_str = str(context.file_path.resolve())
            file_hash_bytes = hashlib.md5(file_path_str.encode()).digest()
            file_hash = int.from_bytes(file_hash_bytes[:8], byteorder='big', signed=False)
            ids = [file_hash + i for i in range(len(context.chunks))]

            self.vector_store.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
        else:
            for chunk in context.chunks:
                self._inject_access_metadata(chunk.metadata, context)
            
            self.vector_store.add_documents(context.chunks)

        self.vector_store.persist()
        logger.info(f"Saved {len(context.chunks)} chunks to vector store")
