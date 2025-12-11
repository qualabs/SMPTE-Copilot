"""PyMuPDF-based PDF loader implementation."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import pymupdf4llm
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader as LangChainPyMuPDFLoader

from ..constants import DEFAULT_ENCODING
from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]


class PyMuPDFLoader:
    """Load PDFs using PyMuPDF and export Markdown representations.

    This is a concrete implementation of the DocumentLoader protocol
    using PyMuPDF and pymupdf4llm libraries.

    The loader receives a configuration dictionary and extracts the necessary
    parameters from it. Supported config keys:
    - file_path (required): Path to the PDF file
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
            If 'file_path' is missing or the file is not a PDF.
        FileNotFoundError
            If the PDF file does not exist.
        """
        self.config = config

        file_path = config.get("file_path")
        if not file_path:
            raise ValueError("'file_path' is required in loader configuration")

        self.pdf_path = Path(file_path).expanduser().resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got file with extension: {self.pdf_path.suffix}"
            )

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    def load_documents(self) -> list[Document]:
        """Load the PDF into LangChain Document objects.

        Returns
        -------
        List of Document objects representing the PDF content.

        Raises
        ------
        Exception
            If the PDF cannot be loaded (e.g., corrupted file, permission issues).
        """
        try:
            loader = LangChainPyMuPDFLoader(str(self.pdf_path))
            return loader.load()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PDF from {self.pdf_path}: {e}"
            ) from e

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        """Return the PDF rendered as Markdown text.

        Parameters
        ----------
        pages
            Optional sequence of page numbers, range, or None for all pages.

        Raises
        ------
        RuntimeError
            If the PDF cannot be converted to Markdown (e.g., corrupted file).
        """
        try:
            return pymupdf4llm.to_markdown(str(self.pdf_path), pages=pages)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert PDF to Markdown from {self.pdf_path}: {e}"
            ) from e

    def to_markdown_file(
        self,
        *,
        pages: PageSpecifier = None,
        output_path: Path | None = None,
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
        """
        md_text = self.to_markdown_text(pages=pages)
        destination = self._resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {destination}")

        destination.write_text(md_text, encoding=DEFAULT_ENCODING)
        return destination

    def _resolve_output_path(self, output_path: Path | None) -> Path:
        """Resolve the output path for the markdown file.

        Parameters
        ----------
        output_path
            Optional explicit output path. If None, generates a default path
            based on the PDF file name in the output directory or PDF's parent directory.
        """
        if output_path is not None:
            return Path(output_path).expanduser().resolve()

        target_dir = self.output_dir or self.pdf_path.parent
        return target_dir / f"{self.pdf_path.stem}.md"


def create_pymupdf_loader(config: dict[str, Any]) -> DocumentLoader:
    """Create a PyMuPDF loader from configuration.

    Parameters
    ----------
    config
        Configuration dictionary. Must contain:
        - file_path (required): Path to the PDF file
        Optional keys:
        - output_dir (optional): Directory for output markdown files
        - Any other keys are stored in the loader's config attribute

    Returns
    -------
    DocumentLoader instance.

    Raises
    ------
    ValueError
        If 'file_path' is missing or the file is not a PDF.
    FileNotFoundError
        If the PDF file does not exist.
    """
    return PyMuPDFLoader(config=config)

