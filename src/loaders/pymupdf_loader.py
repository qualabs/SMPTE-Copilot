from __future__ import annotations

"""PyMuPDF-based PDF loader implementation."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import pymupdf4llm
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader as LangChainPyMuPDFLoader

from .protocol import DocumentLoader

PageSpecifier = Union[Sequence[int], range, None]


class PyMuPDFLoader(DocumentLoader):
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

        self.input_path = Path(file_path).expanduser().resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.input_path}")

        if self.input_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected PDF file, got file with extension: {self.input_path.suffix}"
            )

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    def load_documents(self) -> list[Document]:
        """Load the PDF into LangChain Document objects with markdown-formatted content.

        Returns
        -------
        List of Document objects with markdown in page_content.

        Raises
        ------
        Exception
            If the PDF cannot be loaded (e.g., corrupted file, permission issues).
        """
        try:
            md_text = pymupdf4llm.to_markdown(str(self.input_path))

            # Get metadata using LangChain loader
            loader = LangChainPyMuPDFLoader(str(self.input_path))
            docs = loader.load()

            # Use first document's metadata but replace content with markdown
            metadata = docs[0].metadata if docs else {
                "source": str(self.input_path),
                "file_name": self.input_path.name,
            }
            metadata["loader"] = "PyMuPDFLoader"
            metadata["file_type"] = ".pdf"

            return [
                Document(
                    page_content=md_text,
                    metadata=metadata
                )
            ]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load PDF from {self.input_path}: {e}"
            ) from e

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

