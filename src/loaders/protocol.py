"""Protocol for document loader implementations."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Union

from langchain.schema import Document

PageSpecifier = Union[Sequence[int], range, None]


class DocumentLoader(Protocol):
    """Protocol for document loader implementations.

    Any class implementing these methods can load documents from various
    sources (PDF, DOCX, HTML, etc.) and convert them to a standard format.
    This allows swapping implementations without changing the rest of the code.
    """

    def load_documents(self) -> list[Document]:
        """Load documents into LangChain Document objects."""
        ...

    def to_markdown_text(self, pages: PageSpecifier = None) -> str:
        """Convert the document to Markdown text.

        Parameters
        ----------
        pages
            Optional sequence of page numbers, range, or None for all pages.
        """
        ...

    def to_markdown_file(
        self,
        *,
        pages: PageSpecifier = None,
        output_path: Path | None = None,
        overwrite: bool = True,
    ) -> Path:
        """Save the document as a Markdown file.

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
        ...

