from __future__ import annotations

"""Protocol for document loader implementations."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, Union

from langchain.schema import Document

from ..constants import DEFAULT_ENCODING

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
        md_text: str | None = None,
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
            based on the input file name in the output directory or input file's parent directory.
        """
        if output_path is not None:
            return Path(output_path).expanduser().resolve()

        target_dir = self.output_dir or self.input_path.parent
        return target_dir / f"{self.input_path.stem}.md"

