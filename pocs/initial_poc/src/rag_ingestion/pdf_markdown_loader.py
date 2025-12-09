"""Utilities for converting PDFs into Markdown for RAG ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf4llm

PageSpecifier = Union[Sequence[int], range, None]


@dataclass
class PDFMarkdownLoader:
    """Load PDFs via LangChain and export Markdown representations."""

    pdf_path: Path
    output_dir: Union[Path, None] = None

    def __post_init__(self) -> None:
        self.pdf_path = self.pdf_path.expanduser().resolve()
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

        if self.output_dir is not None:
            self.output_dir = self.output_dir.expanduser().resolve()

    # LangChain integration -------------------------------------------------
    def load_documents(self) -> list:
        """Load the PDF into LangChain Document objects."""
        loader = PyMuPDFLoader(str(self.pdf_path))
        return loader.load()

    # Markdown export -------------------------------------------------------
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

        destination.write_text(md_text, encoding="utf-8")
        return destination

    # Internal helpers ------------------------------------------------------
    def _resolve_output_path(self, output_path: Union[Path, None]) -> Path:
        if output_path is not None:
            return output_path.expanduser().resolve()

        target_dir = self.output_dir or self.pdf_path.parent
        return target_dir / f"{self.pdf_path.stem}.md"


__all__ = ["PDFMarkdownLoader"]

