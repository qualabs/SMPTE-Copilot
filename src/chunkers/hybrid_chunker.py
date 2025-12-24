"""Hybrid chunker implementation using Docling's HybridChunker."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from langchain.schema import Document
from pydantic import ConfigDict

from ..constants import DEFAULT_ENCODING
from .protocol import Chunker

# Try to import Docling components (optional dependency)
try:
    from docling.chunking import HybridChunker as DoclingHybridChunker
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
except ImportError:
    DoclingHybridChunker = None
    InputFormat = None
    DocumentConverter = None
    BaseTokenizer = object  # Fallback if not available

# Try to import new google.genai API
try:
    from google import genai
except ImportError:
    genai = None

# GeminiTokenizer is an internal helper class, defined in this file
# It extends Docling's BaseTokenizer to work with HybridChunker
# Uses the new google.genai API (not the deprecated google.generativeai)


class GeminiTokenizer(BaseTokenizer):
    """Custom tokenizer for Docling's HybridChunker that uses Gemini's token counting API.

    This tokenizer integrates with Google's Gemini API to accurately count tokens
    for text chunks, with a fast local fallback for performance.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any = None
    model: str = "gemini-embedding-001"
    max_tokens: int = 2048

    def __init__(
        self,
        client: Any = None,
        model: str = "gemini-embedding-001",
        max_tokens: int = 2048,
        google_api_key: Optional[str] = None,
    ):
        """Initialize the Gemini tokenizer.

        Parameters
        ----------
        client
            Optional genai.Client instance. If not provided, will be created from API key.
        model
            Gemini model name to use for token counting (default: gemini-embedding-001).
            Note: Some models like "models/embedding-001" don't support countTokens API.
        max_tokens
            Maximum number of tokens per chunk (default: 2048).
        google_api_key
            Optional Google API key. Used only if client is not provided.
            If not provided, uses GOOGLE_API_KEY env var.
        """
        # Initialize client using new google.genai API if not provided
        if client is None and genai is not None:
            try:
                api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
                if api_key:
                    client = genai.Client(api_key=api_key)
            except Exception:
                # API initialization failed, will use fallback
                client = None

        # Initialize BaseTokenizer (Pydantic model) with proper fields
        super().__init__(
            client=client,
            model=model,
            max_tokens=max_tokens,
        )
        
        # Check if model supports countTokens API
        # Some embedding models (like "models/embedding-001") don't support it
        self._use_api = False
        if self.client is not None:
            # Test if countTokens is supported by trying with a small test string
            try:
                test_response = self.client.models.count_tokens(model=self.model, contents="test")
                # If we get here, the API is supported
                self._use_api = True
            except Exception:
                # Model doesn't support countTokens, use approximation
                self._use_api = False
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Model '{self.model}' does not support countTokens API. "
                    "Using fast approximation for token counting."
                )

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's API with fallback to local estimation.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        Number of tokens (from API or estimated).
        """
        # Only use API if it's confirmed to be supported (avoid repeated 404 errors)
        if self._use_api and self.client is not None:
            try:
                response = self.client.models.count_tokens(model=self.model, contents=text)
                return response.total_tokens
            except Exception:
                # API call failed, fall back to approximation
                # Disable API for future calls to avoid repeated failures
                self._use_api = False
                pass

        # Fast local fallback: approximate 4 characters per token
        # This is fast and works for all models, even those without countTokens API
        return len(text) // 4

    def get_max_tokens(self) -> int:
        """Returns the maximum tokens allowed per chunk.

        Returns
        -------
        Maximum tokens.
        """
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        """Returns the underlying tokenizer/client object.

        Returns
        -------
        The genai.Client instance.
        """
        return self.client

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks based on token limits.

        Parameters
        ----------
        text
            Text to split.

        Returns
        -------
        List of text chunks.
        """
        if not text:
            return []

        chunks = []
        words = text.split()
        current_chunk = []

        for word in words:
            # Estimate tokens for current chunk + new word
            test_chunk = " ".join(current_chunk + [word])
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class HybridChunker:
    """Hybrid chunker using Docling's semantic + fixed-size chunking.

    This chunker combines semantic chunking with token-based fixed-size
    chunking, optimized for Gemini models with a 2000 token limit.
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        merge_peers: bool = False,
        google_api_key: Optional[str] = None,
        model: str = "gemini-embedding-001",
    ):
        """Initialize the hybrid chunker.

        Parameters
        ----------
        max_tokens
            Maximum tokens per chunk (default: 2000 for Gemini).
        merge_peers
            Whether to merge peer chunks (default: False).
        google_api_key
            Optional Google API key for token counting. If not provided,
            uses GOOGLE_API_KEY environment variable.
        model
            Gemini model name for token counting (default: models/embedding-001).
        """
        if DoclingHybridChunker is None:
            raise ImportError(
                "docling is required for hybrid chunking. "
                "Install it with `pip install docling` or `pip install docling-core`."
            )

        self.max_tokens = max_tokens
        self.merge_peers = merge_peers

        # Initialize Gemini tokenizer with API key
        # Note: Tokenizer uses "gemini-embedding-001" (without "models/" prefix) for countTokens API
        # Embedding model uses "models/gemini-embedding-001" (with "models/" prefix)
        api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        # Convert model name from embedding format to tokenizer format if needed
        tokenizer_model = model
        if tokenizer_model.startswith("models/"):
            # Remove "models/" prefix for tokenizer (e.g., "models/gemini-embedding-001" -> "gemini-embedding-001")
            tokenizer_model = tokenizer_model.replace("models/", "")
        self.tokenizer = GeminiTokenizer(
            model=tokenizer_model,
            max_tokens=max_tokens,
            google_api_key=api_key
        )

        # Initialize Docling's HybridChunker
        # Note: max_tokens is passed to HybridChunker (as per usage example)
        try:
            self._chunker = DoclingHybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=max_tokens,
                merge_peers=merge_peers,
            )
            # Pre-create a DocumentConverter for efficiency (reuse across chunk operations)
            self._doc_converter = DocumentConverter() if DocumentConverter else None
        except Exception:
            # Fallback if Docling API is different
            self._chunker = None
            self._doc_converter = None
            # We'll implement a fallback chunking strategy

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
        - google_api_key: str (optional) - Google API key for token counting
        - model: str (optional) - Gemini model name for token counting (default: models/embedding-001)

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
    google_api_key = config.get("google_api_key")
    model = config.get("model", "models/embedding-001")

    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got: {max_tokens}")

    return HybridChunker(
        max_tokens=max_tokens,
        merge_peers=merge_peers,
        google_api_key=google_api_key,
        model=model,
    )

