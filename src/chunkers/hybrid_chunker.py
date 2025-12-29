"""Hybrid chunker implementation using Docling's HybridChunker."""
import logging

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document

from ..constants import DEFAULT_ENCODING
from .protocol import Chunker
from .tokenizers import Tokenizer, TokenizerFactory, TokenizerType

from docling.chunking import HybridChunker as DoclingHybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter


class HybridChunker:
    """Hybrid chunker using Docling's semantic + fixed-size chunking.

    This chunker combines semantic chunking with token-based fixed-size
    chunking, optimized for Gemini models with a 2000 token limit.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        merge_peers: bool = False,
        tokenizer: Optional[Tokenizer] = None,
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
        self.logger = logging.getLogger()

        self.logger.info(
            f"Initializing hybrid chunker (max_tokens: {max_tokens}, merge_peers: {merge_peers})..."
        )

        self.max_tokens = max_tokens
        self.merge_peers = merge_peers

        self.tokenizer = tokenizer
        
        self._chunker = DoclingHybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )
        
        self._doc_converter = DocumentConverter()
        
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[Document]:
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

        # Use Docling's HybridChunker if available (semantic + token-based)
        # Note: HybridChunker works with DoclingDocument, so we need to convert text first
        if self._chunker is not None:
            try:
                # Convert text to DoclingDocument for chunking
                if DocumentConverter is None or InputFormat is None:
                    raise ImportError("Docling components not available")
                
                # Use pre-created converter for efficiency (reuse across operations)
                converter = self._doc_converter or DocumentConverter()
                # Convert markdown text to DoclingDocument
                # For chunk_text(), we need to write to a temp file since we only have text
                # But chunk_markdown_file() can use the file path directly (more efficient)
                # Write markdown text to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(text)
                    tmp_path = tmp_file.name
                
                try:
                    # Convert markdown file to DoclingDocument (using file path)
                    # Docling auto-detects format from file extension (.md = markdown)
                    # Pattern: converter.convert(source=str(file_path)) - no input_format parameter needed
                    result = converter.convert(source=tmp_path)
                    dl_doc = result.document
                finally:
                    # Clean up temp file
                    try:
                        Path(tmp_path).unlink()
                    except Exception:
                        pass
                
                # Chunk using Docling's HybridChunker
                # Note: This may be slow for large documents due to semantic processing
                chunks = list(self._chunker.chunk(dl_doc=dl_doc))
                documents = []
                for i, chunk in enumerate(chunks):
                    # Use contextualize() to get enriched text (as per Docling docs)
                    chunk_text = self._chunker.contextualize(chunk=chunk)
                    
                    # Check if chunk exceeds max_tokens and split if needed
                    chunk_tokens = self.tokenizer.count_tokens(chunk_text)
                    if chunk_tokens > self.max_tokens:
                        # Chunk is too large, split it using tokenizer's split_text
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Chunk {i} exceeds max_tokens ({chunk_tokens} > {self.max_tokens}). "
                            "Splitting into smaller chunks."
                        )
                        # Split the oversized chunk
                        sub_chunks = self.tokenizer.split_text(chunk_text)
                        for j, sub_chunk_text in enumerate(sub_chunks):
                            sub_chunk_metadata = {
                                **metadata,
                                "chunk_index": len(documents),
                                "total_chunks": len(chunks) + len(sub_chunks) - 1,  # Approximate
                                "chunking_method": "hybrid_split",
                                "original_chunk_index": i,
                                "sub_chunk_index": j,
                            }
                            documents.append(Document(page_content=sub_chunk_text, metadata=sub_chunk_metadata))
                    else:
                        # Chunk is within limits, use as-is
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunking_method": "hybrid",
                        }
                        documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                
                # Update total_chunks in all documents now that we know the final count
                final_count = len(documents)
                for idx, doc in enumerate(documents):
                    doc.metadata["total_chunks"] = final_count
                    doc.metadata["chunk_index"] = idx
                
                return documents
            except Exception as e:
                # Fallback to tokenizer-based chunking if hybrid chunking fails
                # This is faster but less semantically aware
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Hybrid chunking failed, using fallback: {e}")
                pass

        # Fallback: Use tokenizer-based chunking (simple token splitting)
        # This is faster but less semantically aware than hybrid chunking
        chunk_texts = self.tokenizer.split_text(text)
        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "chunking_method": "hybrid_token_based",
            }
            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents

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
        self, file_path: str, encoding: str = DEFAULT_ENCODING
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

        # Use Docling's HybridChunker if available (semantic + token-based)
        # Use the markdown file directly (no temp file needed - file already exists!)
        if self._chunker is not None:
            try:
                # Convert markdown file to DoclingDocument (using file path directly)
                # This matches the Docling documentation pattern from run_with_formats example
                if DocumentConverter is None or InputFormat is None:
                    raise ImportError("Docling components not available")
                
                converter = self._doc_converter or DocumentConverter()
                # Convert markdown file to DoclingDocument (using file path, not string)
                # Docling auto-detects format from file extension (.md = markdown)
                # Pattern: converter.convert(source=str(file_path)) - no input_format parameter needed
                result = converter.convert(source=str(path))
                dl_doc = result.document
                
                # Chunk using Docling's HybridChunker
                chunks = list(self._chunker.chunk(dl_doc=dl_doc))
                documents = []
                for i, chunk in enumerate(chunks):
                    # Use contextualize() to get enriched text (as per Docling docs)
                    chunk_text = self._chunker.contextualize(chunk=chunk)
                    
                    # Check if chunk exceeds max_tokens and split if needed
                    chunk_tokens = self.tokenizer.count_tokens(chunk_text)
                    if chunk_tokens > self.max_tokens:
                        # Chunk is too large, split it using tokenizer's split_text
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Chunk {i} exceeds max_tokens ({chunk_tokens} > {self.max_tokens}). "
                            "Splitting into smaller chunks."
                        )
                        # Split the oversized chunk
                        sub_chunks = self.tokenizer.split_text(chunk_text)
                        for j, sub_chunk_text in enumerate(sub_chunks):
                            sub_chunk_metadata = {
                                **metadata,
                                "chunk_index": len(documents),
                                "total_chunks": len(chunks) + len(sub_chunks) - 1,  # Approximate
                                "chunking_method": "hybrid_split",
                                "original_chunk_index": i,
                                "sub_chunk_index": j,
                            }
                            documents.append(Document(page_content=sub_chunk_text, metadata=sub_chunk_metadata))
                    else:
                        # Chunk is within limits, use as-is
                        chunk_metadata = {
                            **metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "chunking_method": "hybrid",
                        }
                        documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))
                
                # Update total_chunks in all documents now that we know the final count
                final_count = len(documents)
                for idx, doc in enumerate(documents):
                    doc.metadata["total_chunks"] = final_count
                    doc.metadata["chunk_index"] = idx
                
                return documents
            except Exception as e:
                # Fallback to tokenizer-based chunking if hybrid chunking fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Hybrid chunking failed, using fallback: {e}")
                # Fall through to tokenizer-based chunking

        # Fallback: Use tokenizer-based chunking (simple token splitting)
        # Read text and use tokenizer-based chunking
        text = path.read_text(encoding=encoding)
        chunk_texts = self.tokenizer.split_text(text)
        documents = []
        for i, chunk_text in enumerate(chunk_texts):
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunk_texts),
                "chunking_method": "hybrid_token_based",
            }
            documents.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return documents


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

