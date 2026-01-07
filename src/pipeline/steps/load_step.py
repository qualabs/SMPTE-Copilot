from __future__ import annotations

"""Step that loads a document and converts it to markdown."""

import logging

from ...loaders.protocol import DocumentLoader
from ..contexts.ingestion_context import IngestionContext


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
        logger = logging.getLogger(__name__)
        logger.info(f"Loading document: {context.file_path}")

        loaded_docs = self.loader.load_documents()

        if loaded_docs:
            context.metadata = loaded_docs[0].metadata.copy()

        md_text = self.loader.to_markdown_text()
        context.raw_text = md_text

        markdown_path = self.loader.to_markdown_file(md_text=md_text)
        context.markdown_path = markdown_path

        logger.info(f"Markdown saved to: {markdown_path}")
