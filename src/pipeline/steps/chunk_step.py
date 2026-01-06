from __future__ import annotations

"""Step that chunks the markdown text into smaller pieces."""

import logging

from ...chunkers.protocol import Chunker
from ..contexts.ingestion_context import IngestionContext


class ChunkStep:
    """Step that chunks the markdown text into smaller pieces."""

    def __init__(self, chunker: Chunker):
        """Initialize the chunk step.

        Parameters
        ----------
        chunker
            Chunker instance created by ChunkerFactory.
        """
        self.chunker = chunker

    def run(self, context: IngestionContext) -> None:
        """Chunk the markdown file into documents.

        Parameters
        ----------
        context
            Ingestion context with markdown_path set.
        """
        logger = logging.getLogger(__name__)
        if not context.markdown_path:
            context.mark_failed("Markdown path not set. Load step must run first.")
            return

        logger.info(f"Chunking markdown file: {context.markdown_path}")

        base_metadata = context.metadata.copy() if context.metadata else {}

        chunks = self.chunker.chunk_markdown_file(str(context.markdown_path))

        for chunk in chunks:
            chunk.metadata.update(base_metadata)

        context.chunks = chunks

        logger.info(f"Created {len(chunks)} chunks")
