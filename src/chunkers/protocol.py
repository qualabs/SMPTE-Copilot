"""Protocol for chunker implementations."""
from __future__ import annotations

from typing import Protocol

from langchain.schema import Document

from ..constants import DEFAULT_ENCODING


class Chunker(Protocol):
    """Protocol for text chunking implementations.

    Any class implementing these methods can chunk documents using
    different strategies (recursive, character-based, token-based, semantic, etc.).
    This allows swapping chunking algorithms without changing the rest of the code.
    """

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """Chunk a text string into Document objects.

        Parameters
        ----------
        text
            The text to chunk.
        metadata
            Optional metadata to add to each chunk.

        Returns
        -------
        List of Document objects, each representing a chunk.
        """
        ...

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of Document objects into smaller chunks.

        Parameters
        ----------
        documents
            List of Document objects to chunk.

        Returns
        -------
        List of chunked Document objects.
        """
        ...

    def chunk_markdown_file(
        self,
        file_path: str,
        encoding: str = DEFAULT_ENCODING
    ) -> list[Document]:
        """Load a markdown file and chunk it.

        Parameters
        ----------
        file_path
            Path to the markdown file.
        encoding
            File encoding (default: utf-8).

        Returns
        -------
        List of chunked Document objects.
        """
        ...

