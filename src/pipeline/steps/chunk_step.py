"""Step that chunks the markdown text into smaller pieces."""
from __future__ import annotations

import logging

from ...chunkers.protocol import Chunker
from ..contexts.ingestion_context import IngestionContext
from ..step import PipelineStep


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
        logger = logging.getLogger()
        if not context.markdown_path:
            context.mark_failed("Markdown path not set. Load step must run first.")
            return

        logger.info(f"Chunking markdown file: {context.markdown_path}")

        chunks = self.chunker.chunk_markdown_file(str(context.markdown_path))
        context.chunks = chunks

        logger.info(f"Created {len(chunks)} chunks")
