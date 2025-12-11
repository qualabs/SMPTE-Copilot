"""Step that loads a document and converts it to markdown."""
from __future__ import annotations

import logging

from ...loaders.protocol import DocumentLoader
from ..contexts.ingestion_context import IngestionContext
from ..step import PipelineStep


class LoadStep:
    """Step that loads a document and converts it to markdown."""

    def __init__(self, loader: DocumentLoader):
        """Initialize the load step.

        Parameters
        ----------
        loader
            Document loader instance created by LoaderFactory.
        """
        self.loader = loader

    def run(self, context: IngestionContext) -> None:
        """Load the document and save as markdown.

        Parameters
        ----------
        context
            Ingestion context with file_path set.
        """
        logger = logging.getLogger()
        logger.info(f"Loading document: {context.file_path}")

        markdown_path = self.loader.to_markdown_file()
        context.markdown_path = markdown_path
        context.raw_text = self.loader.to_markdown_text()

        logger.info(f"Markdown saved to: {markdown_path}")
