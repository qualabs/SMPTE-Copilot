from __future__ import annotations

import logging
import re
from typing import Any, Optional

from google import genai
from langchain_core.documents import Document

from .constants import DEFAULT_MAX_RERANK_CHARS, DEFAULT_RERANK_MODEL
from .protocol import Reranker


class GeminiReranker(Reranker):
    """Reranker backed by Gemini API for relevance scoring."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: Optional[str] = None,
        max_chars: int = DEFAULT_MAX_RERANK_CHARS,
    ):
        """Initialize Gemini reranker.

        Parameters
        ----------
        model
            Gemini model name for reranking
        api_key
            Google API key (if None, uses environment variable)
        max_chars
            Maximum characters of document content to send for scoring
        """
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.max_chars = max_chars
        self.logger = logging.getLogger(__name__)

    def rerank(
        self, query: str, documents: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """Rerank documents using Gemini to score relevance.

        Parameters
        ----------
        query
            User's search query
        documents
            List of (Document, score) tuples from initial retrieval

        Returns
        -------
        List of (Document, score) tuples reordered by relevance with new scores
        """
        if not documents:
            return documents

        self.logger.info(f"Reranking {len(documents)} documents with Gemini")

        reranked = []
        for doc, _original_score in documents:
            # Truncate document content if too long
            content = doc.page_content[: self.max_chars]
            if len(doc.page_content) > self.max_chars:
                content += "...[truncated]"

            # Score the document relevance
            score = self._score_document(query, content)
            reranked.append((doc, score))

        # Sort by new scores (higher is better)
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info("Reranking completed")
        return reranked

    def _score_document(self, query: str, document_content: str) -> float:
        """Score a single document's relevance to the query.

        Parameters
        ----------
        query
            User's search query
        document_content
            Document text content (possibly truncated)

        Returns
        -------
        Relevance score from 0.0 to 10.0 (higher is more relevant)
        """
        prompt = f"""Rate the relevance of this document to the query on a scale of 0-10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant and directly answers the query

Query: {query}

Document:
{document_content}

Output ONLY a single number between 0 and 10 (can include decimals like 7.5). Do not include any explanation."""

        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            score_text = (resp.text or "0").strip()

            match = re.search(r"\d+\.?\d*", score_text)
            if match:
                score = float(match.group())
                # Clamp to valid range
                return max(0.0, min(10.0, score))
            else:
                self.logger.warning(f"Could not parse score from: {score_text}")
                return 0.0

        except Exception as e:
            self.logger.info(f"Error scoring document: {e}")
            return 0.0


def create_gemini_reranker(config: dict[str, Any]) -> Reranker:
    """Factory function to create a Gemini reranker from config dict.

    Parameters
    ----------
    config
        Configuration dictionary with optional keys: model, api_key, max_chars

    Returns
    -------
    Configured GeminiReranker instance
    """
    return GeminiReranker(
        model=config.get("model", DEFAULT_RERANK_MODEL),
        api_key=config.get("api_key"),
        max_chars=config.get("max_chars", DEFAULT_MAX_RERANK_CHARS),
    )
