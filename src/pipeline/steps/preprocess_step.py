from __future__ import annotations

"""Step that preprocesses markdown text to remove repeated content."""

import logging

from ...constants import DEFAULT_ENCODING
from ...preprocessing.protocol import Preprocessor
from ..contexts.ingestion_context import IngestionContext


class PreprocessStep:
    """Step that preprocesses markdown text to remove repeated content.

    This step detects and removes repeated headers, footers, and page numbers
    that can pollute chunks and reduce retrieval precision.
    """

    def __init__(self, preprocessor: Preprocessor):
        """Initialize the preprocessing step.

        Parameters
        ----------
        preprocessor
            Preprocessor instance created by PreprocessorFactory.
        """
        self.preprocessor = preprocessor

    def run(self, context: IngestionContext) -> None:
        """Preprocess the markdown text to remove repeated content.

        Parameters
        ----------
        context
            Ingestion context with raw_text and markdown_path set.
        """
        logger = logging.getLogger(__name__)
        if not context.raw_text:
            context.mark_failed("Raw text not set. Load step must run first.")
            return

        logger.info("Preprocessing markdown to remove repeated content...")

        cleaned_text = self.preprocessor.preprocess(context.raw_text)

        context.raw_text = cleaned_text

        if context.markdown_path:
            context.markdown_path.write_text(
                cleaned_text, encoding=DEFAULT_ENCODING
            )
            logger.info(f"Updated markdown file: {context.markdown_path}")

        logger.info("Preprocessing complete")

