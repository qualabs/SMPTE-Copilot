"""LangChain-based chunker implementation."""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain.schema import Document

from .protocol import Chunker


class LangChainChunker:
    """Chunk markdown text or LangChain documents using LangChain splitters.
    
    This is a concrete implementation of the Chunker protocol using
    LangChain's text splitter implementations.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        method: str = "recursive",
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
        """Create the appropriate text splitter based on method."""
        splitters = {
            "recursive": (RecursiveCharacterTextSplitter, {"separators": ["\n\n", "\n", ". ", " ", ""]}),
            "character": (CharacterTextSplitter, {"separator": "\n\n"}),
            "token": (TokenTextSplitter, {}),
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

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Chunk a markdown text string into LangChain Documents.

        Parameters
        ----------
        text
            The markdown text to chunk.
        metadata
            Optional metadata to add to each chunk.

        Returns
        -------
        List of LangChain Document objects, each representing a chunk.
        """
        if not text or not text.strip():
            return []

        # Create a temporary document to split
        temp_doc = Document(page_content=text, metadata=metadata or {})
        chunks = self._splitter.split_documents([temp_doc])

        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk a list of LangChain Documents into smaller chunks.

        Parameters
        ----------
        documents
            List of LangChain Document objects to chunk.

        Returns
        -------
        List of chunked LangChain Document objects.
        """
        if not documents:
            return []

        # Split all documents
        all_chunks = self._splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        return all_chunks

    def chunk_markdown_file(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
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


def create_langchain_chunker(config: dict) -> Chunker:
    return LangChainChunker(
        chunk_size=config.get("chunk_size"),
        chunk_overlap=config.get("chunk_overlap"),
        method=config.get("method"),
    )