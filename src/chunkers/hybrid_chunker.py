from __future__ import annotations

"""Hybrid chunker implementation using Docling's HybridChunker."""
import logging
import tempfile
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker as DoclingHybridChunker
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document

from ..constants import DEFAULT_ENCODING
from .protocol import Chunker
from .tokenizers import Tokenizer, TokenizerFactory, TokenizerType


class HybridChunker:
    """Hybrid chunker using Docling's semantic + fixed-size chunking.

    This chunker combines semantic chunking with token-based fixed-size
    chunking, optimized for Gemini models with a 2000 token limit.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        merge_peers: bool = False,
        tokenizer: Tokenizer | None = None,
    ):
        """Initialize the hybrid chunker.

        Parameters
        ----------
        max_tokens
            Maximum tokens per chunk (default: 2000).
        merge_peers
            Whether to merge peer chunks (default: False).
        tokenizer
            Optional tokenizer instance. If not provided, uses a simple approximation tokenizer.
        """
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Initializing hybrid chunker (max_tokens: {max_tokens}, merge_peers: {merge_peers})..."
        )

        self.max_tokens = max_tokens
        self.merge_peers = merge_peers

        self.tokenizer = tokenizer

        self.chunker = DoclingHybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )

        self.doc_converter = DocumentConverter()

    def _process_chunks(self, dl_doc: Any, metadata: dict) -> list[Document]:
        """Process Docling chunks into LangChain Documents.

        Parameters
        ----------
        dl_doc
            DoclingDocument to chunk.
        metadata
            Base metadata to attach to all chunks.

        Returns
        -------
        List of chunked Document objects.
        """
        chunks = list(self.chunker.chunk(dl_doc=dl_doc))
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_text = self.chunker.contextualize(chunk=chunk)
            chunk_tokens = self.tokenizer.count_tokens(chunk_text)

            self.logger.info(f"Chunk {i} has {chunk_tokens} tokens, max_tokens: {self.max_tokens}")

            if chunk_tokens > self.max_tokens:
                sub_chunks = self.tokenizer.split_text(chunk_text)
                for j, sub_chunk_text in enumerate(sub_chunks):
                    sub_chunk_metadata = {
                        **metadata,
                        "chunk_index": len(documents),
                        "chunking_method": "hybrid_split",
                        "original_chunk_index": i,
                        "sub_chunk_index": j,
                    }
                    documents.append(Document(page_content=sub_chunk_text, metadata=sub_chunk_metadata))
            else:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": len(documents),
                    "chunking_method": "hybrid",
                }
                documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        final_count = len(documents)
        for idx, doc in enumerate(documents):
            doc.metadata["total_chunks"] = final_count
            doc.metadata["chunk_index"] = idx

        return documents

    def chunk_text(self, text: str, metadata: dict | None = None) -> list[Document]:
        """Chunk text using hybrid chunking strategy.

        Parameters
        ----------
        text
            Text to chunk.
        metadata
            Optional metadata to attach to chunks.

        Returns
        -------
        List of chunked Document objects.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(text)
            tmp_path = tmp_file.name

        try:
            result = self.doc_converter.convert(source=tmp_path)
            dl_doc = result.document
        finally:
            Path(tmp_path).unlink()

        return self._process_chunks(dl_doc, metadata)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Chunk a list of Document objects using hybrid chunking.

        Parameters
        ----------
        documents
            List of Document objects to chunk.

        Returns
        -------
        List of chunked Document objects.
        """
        if not documents:
            return []

        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.page_content, metadata=doc.metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def chunk_markdown_file(
        self,
        file_path: str,
        encoding: str = DEFAULT_ENCODING,
    ) -> list[Document]:
        """Load a markdown file and chunk it using hybrid chunking.

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

        metadata = {
            "source": str(path),
            "file_name": path.name,
        }

        result = self.doc_converter.convert(source=str(path))
        dl_doc = result.document

        return self._process_chunks(dl_doc, metadata)



def create_hybrid_chunker(config: dict[str, Any]) -> Chunker:
    """Create a hybrid chunker from configuration.

    Parameters
    ----------
    config
        Configuration dictionary with keys:
        - max_tokens: int (optional) - Maximum tokens per chunk (default: 2000)
        - merge_peers: bool (optional) - Whether to merge peer chunks (default: False)
        - tokenizer: str (optional) - Tokenizer type (simple, gemini). Default: simple
        - tokenizer_config: dict (optional) - Tokenizer-specific configuration.
          For gemini: google_api_key, model
          For simple: no additional config needed

    Returns
    -------
    Chunker instance.

    Raises
    ------
    ValueError
        If invalid configuration values are provided.
    """
    max_tokens = config.get("max_tokens", 2000)
    merge_peers = config.get("merge_peers", False)
    tokenizer_type_str = config.get("tokenizer", "simple")
    tokenizer_config = config.get("tokenizer_config", {})

    tokenizer_type = TokenizerType(tokenizer_type_str)
    tokenizer_config_with_max = {**tokenizer_config, "max_tokens": max_tokens}
    tokenizer = TokenizerFactory.create(tokenizer_type, **tokenizer_config_with_max)

    return HybridChunker(
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        tokenizer=tokenizer,
    )
