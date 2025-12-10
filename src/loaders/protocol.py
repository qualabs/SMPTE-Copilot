"""Protocol for document loader implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain.schema import Document


class DocumentLoader(Protocol):
    """Protocol for document loader implementations.

    Any class implementing these methods can load documents from various
    sources (PDF, DOCX, HTML, etc.) and convert them to a standard format.
    This allows swapping implementations without changing the rest of the code.
    """

    def load_documents(self) -> list[Document]:
        """Load documents into LangChain Document objects.

        Returns
        -------
        List of Document objects representing the loaded content.
        """
        ...

    def to_markdown_text(self, pages: list[int] | None = None) -> str:
        """Convert the document to Markdown text.

        Parameters
        ----------
        pages
            Optional list of page numbers to extract (for multi-page documents).
            If None, extracts all pages.

        Returns
        -------
        Markdown representation of the document.
        """
        ...

    def to_markdown_file(
        self,
        *,
        pages: list[int] | None = None,
        output_path: Path | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Save the document as a Markdown file.

        Parameters
        ----------
        pages
            Optional list of page numbers to extract.
        output_path
            Optional path where to save the Markdown file.
            If None, uses a default path based on the source document.
        overwrite
            Whether to overwrite existing files.

        Returns
        -------
        Path to the saved Markdown file.
        """
        ...

