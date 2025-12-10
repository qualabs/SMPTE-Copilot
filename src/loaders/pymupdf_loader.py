"""PyMuPDF-based PDF loader implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union, List, Dict, Any

from langchain_community.document_loaders import PyMuPDFLoader as LangChainPyMuPDFLoader
from langchain.schema import Document
import pymupdf4llm

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

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the loader with a configuration dictionary.
        
        Parameters
        ----------
        config
            Configuration dictionary. Must contain 'file_path' key.
            Optional keys: 'output_dir', and any other loader-specific config.
        """
        self.config = config
        
        file_path = config.get("file_path")
        
        self.pdf_path = Path(file_path).expanduser().resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        output_dir = config.get("output_dir")
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    def load_documents(self) -> List[Document]:
        """Load the PDF into LangChain Document objects."""
        loader = LangChainPyMuPDFLoader(str(self.pdf_path))
        return loader.load()

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        """Return the PDF rendered as Markdown text."""
        return pymupdf4llm.to_markdown(str(self.pdf_path), pages=pages)

    def to_markdown_file(
        self,
        *,
        pages: PageSpecifier = None,
        output_path: Union[Path, None] = None,
        overwrite: bool = True,
    ) -> Path:
        """Persist the rendered Markdown to disk and return its path."""
        md_text = self.to_markdown_text(pages=pages)
        destination = self._resolve_output_path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {destination}")

        destination.write_text(md_text, encoding=DEFAULT_ENCODING)
        return destination

    def _resolve_output_path(self, output_path: Union[Path, None]) -> Path:
        """Resolve the output path for the markdown file."""
        if output_path is not None:
            return Path(output_path).expanduser().resolve()

        target_dir = self.output_dir or self.pdf_path.parent
        return target_dir / f"{self.pdf_path.stem}.md"


def create_pymupdf_loader(config: Dict[str, Any]) -> DocumentLoader:
    """Create a PyMuPDF loader from configuration.
    
    Parameters
    ----------
    config
        Configuration dictionary. The loader will extract necessary parameters
        from this config. Supported keys:
        - file_path, pdf_path, or path (required): Path to the PDF file
        - output_dir, output_directory, or markdown_dir (optional): Directory for output
        - Any other keys are passed through to the loader
    
    Returns
    -------
    DocumentLoader instance.
    """
    return PyMuPDFLoader(config=config)

