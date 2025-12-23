"""Docling-based document loader implementation."""
from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, Union

from langchain.schema import Document

from ..constants import DEFAULT_ENCODING
from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]

# Suppress RapidOCR warnings globally (harmless - just means no text found in images)
# This is normal for text-based PDFs that don't need OCR
# The warnings come from RapidOCR's internal code (not our code)
# The "main.py:125" reference is from RapidOCR's internal main.py file, not our code
rapidocr_logger = logging.getLogger("RapidOCR")
rapidocr_logger.setLevel(logging.ERROR)  # Only show errors, not warnings

# Also suppress warnings module warnings from RapidOCR (in case they use warnings.warn)
warnings.filterwarnings("ignore", message=".*RapidOCR.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*text detection result is empty.*", category=UserWarning)


class DoclingLoader:
    """Load documents using Docling and export Markdown representations.

    This is a concrete implementation of the DocumentLoader protocol
    using IBM's Docling library for parsing PDF, DOCX, and Markdown files.

    The loader receives a configuration dictionary and extracts the necessary
    parameters from it. Supported config keys:
    - file_path (required): Path to the document file (PDF, DOCX, or Markdown)
    - output_dir (optional): Directory for output markdown files
    - Any other keys are stored and can be accessed via self.config
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the loader with a configuration dictionary.

        Parameters
        ----------
        config
            Configuration dictionary. Must contain 'file_path' key.
            Optional keys: 'output_dir', and any other loader-specific config.

        Raises
        ------
        ValueError
            If 'file_path' is missing or the file type is not supported.
        FileNotFoundError
            If the document file does not exist.
        ImportError
            If docling is not installed.
        """
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
        except ImportError as exc:
            raise ImportError(
                "docling is required for Docling loader. "
                "Install it with `pip install docling` or `pip install docling-core`."
            ) from exc

        self.config = config

        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        self.doc_path = Path(file_path).expanduser().resolve()
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Document not found: {self.doc_path}")

        # Check if file type is supported
        supported_extensions = {".pdf", ".docx", ".md", ".markdown"}
        if self.doc_path.suffix.lower() not in supported_extensions:
            raise ValueError(
                f"Unsupported file type: {self.doc_path.suffix}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

        # RapidOCR warnings are already suppressed at module level
        
        # Initialize Docling converter with StandardPdfPipeline and PyPdfiumDocumentBackend
        # This matches the working configuration from the other project
        # Uses default options which include: OCR enabled, table detection enabled, layout detection enabled
        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            
            # Try to import the format options and pipeline classes
            # These may be in different locations depending on docling version
            try:
                from docling.datamodel.pipeline_options import (
                    PdfFormatOption,
                    WordFormatOption,
                )
                from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
                from docling.backend.pypdfium_backend import PyPdfiumDocumentBackend
                from docling.pipeline.simple_pipeline import SimplePipeline
                
                # Define allowed formats
                allowed_formats = [InputFormat.PDF, InputFormat.DOCX, InputFormat.MD]
                
                # Create format options matching the working configuration
                # StandardPdfPipeline with PyPdfiumDocumentBackend provides:
                # - OCR enabled (do_ocr = True)
                # - Table detection enabled (do_table_structure = True)
                # - Layout detection enabled
                # - Auto-selects OCR engine
                format_options = {
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        backend=PyPdfiumDocumentBackend
                    ),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_cls=SimplePipeline
                    )
                }
                
                self.converter = DocumentConverter(
                    allowed_formats=allowed_formats,
                    format_options=format_options
                )
            except ImportError:
                # If specific classes aren't available, use default converter
                # Default DocumentConverter should have OCR and table detection enabled
                # (matching the user's working configuration which uses defaults)
                self.converter = DocumentConverter()
        except Exception as e:
            # Fallback: use default converter if all options fail
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Failed to initialize DocumentConverter with custom options: {e}. "
                "Using default converter."
            )
            self.converter = DocumentConverter()

    def load_documents(self) -> list[Document]:
        """Load the document into LangChain Document objects.

        Returns
        -------
        List of Document objects representing the document content.

        Raises
        ------
        RuntimeError
            If the document cannot be loaded (e.g., corrupted file, permission issues).
        """
        try:
            # Set environment variables to use headless OpenCV if needed
            import os
            import logging
            os.environ.setdefault("OPENCV_HEADLESS", "1")
            # Suppress OpenCV GUI warnings
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            
            # Suppress RapidOCR warnings (harmless - just means no text found in images)
            # This is normal for text-based PDFs that don't need OCR
            rapidocr_logger = logging.getLogger("RapidOCR")
            rapidocr_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
            
            # Convert document using Docling
            result = self.converter.convert(str(self.doc_path))
            
            # Export to markdown
            markdown_text = result.document.export_to_markdown()
            
            # Create LangChain Document
            metadata = {
                "source": str(self.doc_path),
                "file_name": self.doc_path.name,
                "file_type": self.doc_path.suffix.lower(),
            }
            
            return [Document(page_content=markdown_text, metadata=metadata)]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load document from {self.doc_path}: {e}"
            ) from e

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        """Return the document rendered as Markdown text.

        Parameters
        ----------
        pages
            Optional sequence of page numbers, range, or None for all pages.
            Note: Page filtering may not be supported for all formats.

        Returns
        -------
        Markdown text representation of the document.

        Raises
        ------
        RuntimeError
            If the document cannot be converted to Markdown (e.g., corrupted file).
        """
        try:
            # Set environment variables to use headless OpenCV if needed
            import os
            import logging
            os.environ.setdefault("OPENCV_HEADLESS", "1")
            # Suppress OpenCV GUI warnings
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            
            # Suppress RapidOCR warnings (harmless - just means no text found in images)
            # This is normal for text-based PDFs that don't need OCR
            rapidocr_logger = logging.getLogger("RapidOCR")
            rapidocr_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
            
            result = self.converter.convert(str(self.doc_path))
            markdown_text = result.document.export_to_markdown()
            
            # If pages are specified and it's a PDF, filter pages
            # Note: Docling may handle this differently, adjust as needed
            if pages is not None and self.doc_path.suffix.lower() == ".pdf":
                # Docling doesn't directly support page filtering in export_to_markdown
                # This would require custom implementation if needed
                pass
            
            return markdown_text
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert document to Markdown from {self.doc_path}: {e}"
            ) from e

    def to_markdown_file(
        self,
        *,
        pages: PageSpecifier = None,
        output_path: Optional[Path] = None,
        overwrite: bool = True,
    ) -> Path:
        """Persist the rendered Markdown to disk and return its path.

        Parameters
        ----------
        pages
            Optional sequence of page numbers, range, or None for all pages.
        output_path
            Optional path where to save the Markdown file.
            If None, uses a default path based on the source document.
        overwrite
            Whether to overwrite existing files.

        Returns
        -------
        Path to the saved Markdown file.
        """
        md_text = self.to_markdown_text(pages=pages)
        destination = self._resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {destination}")

        destination.write_text(md_text, encoding=DEFAULT_ENCODING)
        return destination

    def _resolve_output_path(self, output_path: Optional[Path]) -> Path:
        """Resolve the output path for the markdown file.

        Parameters
        ----------
        output_path
            Optional explicit output path. If None, generates a default path
            based on the document file name in the output directory or document's parent directory.
        """
        if output_path is not None:
            return Path(output_path).expanduser().resolve()

        target_dir = self.output_dir or self.doc_path.parent
        return target_dir / f"{self.doc_path.stem}.md"


def create_docling_loader(config: dict[str, Any]) -> DocumentLoader:
    """Create a Docling loader from configuration.

    Parameters
    ----------
    config
        Configuration dictionary. Must contain:
        - file_path (required): Path to the document file (PDF, DOCX, or Markdown)
        Optional keys:
        - output_dir (optional): Directory for output markdown files
        - Any other keys are stored in the loader's config attribute

    Returns
    -------
    DocumentLoader instance.

    Raises
    ------
    ValueError
        If 'file_path' is missing or the file type is not supported.
    FileNotFoundError
        If the document file does not exist.
    ImportError
        If docling is not installed.
    """
    return DoclingLoader(config=config)

