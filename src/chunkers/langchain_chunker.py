"""LangChain-based chunker implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)

from ..constants import DEFAULT_ENCODING
from .constants import (
    CHUNK_INDEX_METADATA_KEY,
    CHUNKING_METHOD_CHARACTER,
    CHUNKING_METHOD_RECURSIVE,
    CHUNKING_METHOD_TOKEN,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RECURSIVE_SEPARATORS,
    TOTAL_CHUNKS_METADATA_KEY,
)
from .protocol import Chunker


class LangChainChunker:
    """Chunk markdown text or LangChain documents using LangChain splitters.

    This is a concrete implementation of the Chunker protocol using
    LangChain's text splitter implementations.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        method: str = CHUNKING_METHOD_RECURSIVE,
    ):
        """Initialize the chunker.

        Parameters
        ----------
        chunk_size
            Maximum size of each chunk (characters or tokens depending on method).
        chunk_overlap
            Number of characters/tokens to overlap between chunks.
        method
            Chunking method: "recursive" (default), "character", or "token".
            - "recursive": Smart splitting that tries to preserve structure
            - "character": Simple character-based splitting
            - "token": Token-based splitting (requires tiktoken)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method
        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """Create the appropriate text splitter based on the chunking method.

        Returns
        -------
        Text splitter instance (RecursiveCharacterTextSplitter, CharacterTextSplitter,
        or TokenTextSplitter) configured with chunk_size and chunk_overlap.

        Raises
        ------
        ValueError
            If the method is not one of the supported methods.
        """
        splitters = {
            CHUNKING_METHOD_RECURSIVE: (
                RecursiveCharacterTextSplitter,
                {"separators": RECURSIVE_SEPARATORS},
            ),
            CHUNKING_METHOD_CHARACTER: (CharacterTextSplitter, {"separator": "\n\n"}),
            CHUNKING_METHOD_TOKEN: (TokenTextSplitter, {}),
        }
        if self.method not in splitters:
            raise ValueError(
                f"Unknown method: {self.method}. Choose from: {', '.join(splitters.keys())}"
            )
        splitter_class, extra_kwargs = splitters[self.method]
        return splitter_class(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **extra_kwargs
        )

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
        """Chunk a markdown text string into LangChain Documents."""
        if not text or not text.strip():
            return []

        temp_doc = Document(page_content=text, metadata=metadata or {})
        chunks = self._splitter.split_documents([temp_doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata[CHUNK_INDEX_METADATA_KEY] = i
            chunk.metadata[TOTAL_CHUNKS_METADATA_KEY] = len(chunks)

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of LangChain Documents into smaller chunks."""
        if not documents:
            return []

        all_chunks = self._splitter.split_documents(documents)

        for i, chunk in enumerate(all_chunks):
            chunk.metadata[CHUNK_INDEX_METADATA_KEY] = i
            chunk.metadata[TOTAL_CHUNKS_METADATA_KEY] = len(all_chunks)

        return all_chunks

    def chunk_markdown_file(
        self, file_path: str, encoding: str = DEFAULT_ENCODING
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
        List of chunked LangChain Document objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        text = path.read_text(encoding=encoding)
        metadata = {
            "source": str(path),
            "file_name": path.name,
        }

        return self.chunk_text(text, metadata=metadata)


def create_langchain_chunker(config: dict[str, Any]) -> Chunker:
    """Create a LangChain chunker from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - chunk_size: int (optional) - Size of text chunks in characters
        - chunk_overlap: int (optional) - Overlap between chunks in characters
        - method: str (optional) - Chunking method (recursive, character, token)

    Returns
    -------
    Chunker instance.
    """
    chunk_size = config.get("chunk_size", DEFAULT_CHUNK_SIZE)
    chunk_overlap = config.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    method = config.get("method", CHUNKING_METHOD_RECURSIVE)

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer, got: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    if method not in [CHUNKING_METHOD_RECURSIVE, CHUNKING_METHOD_CHARACTER, CHUNKING_METHOD_TOKEN]:
        raise ValueError(
            f"method must be one of: {CHUNKING_METHOD_RECURSIVE}, {CHUNKING_METHOD_CHARACTER}, "
            f"{CHUNKING_METHOD_TOKEN}, got: {method}"
        )

    return LangChainChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        method=method,
    )
