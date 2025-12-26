"""Protocol for chunker implementations."""
from __future__ import annotations

from typing import Optional, Protocol

from langchain.schema import Document

from ..constants import DEFAULT_ENCODING


class Chunker(Protocol):
    """Protocol for text chunking implementations.

    Any class implementing these methods can chunk documents using
    different strategies (recursive, character-based, token-based, semantic, etc.).
    This allows swapping chunking algorithms without changing the rest of the code.
    """

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """Chunk a text string into Document objects."""
        ...

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of Document objects into smaller chunks."""
        ...

    def chunk_markdown_file(
        self,
        file_path: str,
        encoding: str = DEFAULT_ENCODING
    ) -> list[Document]:
        """Load a markdown file and chunk it."""
        ...

