from __future__ import annotations

import logging
import re
from typing import Any

from google import genai
from google.genai import types
from langchain_core.documents import Document

from .constants import (
    DEFAULT_MAX_RERANK_CHARS,
    DEFAULT_RERANK_MODEL,
    DEFAULT_SCORING_PROMPT,
)
from .protocol import Reranker


class GeminiReranker(Reranker):
    """Reranker backed by Gemini API for relevance scoring."""

    def __init__(
        self,
        model: str = DEFAULT_RERANK_MODEL,
        api_key: str | None = None,
        max_chars: int = DEFAULT_MAX_RERANK_CHARS,
        scoring_prompt: str | None = None,
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
        scoring_prompt
            Custom prompt template for scoring (use {query} and {document} placeholders)
        """
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.max_chars = max_chars
        self.scoring_prompt = scoring_prompt or DEFAULT_SCORING_PROMPT
        self.logger = logging.getLogger(__name__)

    def create_scoring_prompt(self, query: str, document_content: str) -> str:
        """Generate the scoring prompt for document relevance evaluation.

        Parameters
        ----------
        query
            User's search query
        document_content
            Document text to evaluate (may include metadata)

        Returns
        -------
        Formatted prompt string for LLM scoring
        """
        return self.scoring_prompt.format(query=query, document=document_content)

    def rerank(
        self, query: str,
        documents: list[tuple[Document, float]]
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

        self.logger.info(f"🔄 Reranking {len(documents)} documents with Gemini")

        reranked = []
        successful_scores = 0
        zero_scores = 0

        for i, (doc, _original_score) in enumerate(documents, 1):
            self.logger.info(f"--- Document {i}/{len(documents)} ---")

            content = doc.page_content[: self.max_chars]
            if len(doc.page_content) > self.max_chars:
                content += "...[truncated]"

            # Add metadata context if available
            metadata = doc.metadata or {}

            source = metadata.get("source")

            if source:
                content = f"[METADATA]\n{source}\n[CONTENT]\n{content}"

            # Score the document relevance
            score = self._score_document(query, content)
            reranked.append((doc, score))

            if score > 0.0:
                successful_scores += 1
            else:
                zero_scores += 1

        # Sort by new scores (higher is better)
        reranked.sort(key=lambda x: x[1], reverse=True)

        self.logger.info(
            f"✅ Reranking completed: {successful_scores} docs scored > 0, "
            f"{zero_scores} docs scored 0"
        )
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
        doc_preview = document_content[:150].replace("\n", " ")
        self.logger.info(f"📝 Evaluating: {doc_preview}...")

        prompt = self.create_scoring_prompt(query, document_content)

        try:

            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,
            )

            resp = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            # Check if response is None or empty
            if not resp or not resp.text:
                self.logger.info(
                    f"LLM returned empty response for model {self.model}. "
                    "This might be a model availability issue. Returning 0.0"
                )
                return 0.0

            score_text = resp.text.strip()

            self.logger.debug(f"LLM prompt: {prompt}")
            self.logger.debug(f"LLM score: {score_text}")

            # Try to extract a number from the response
            # First try to parse the whole thing as a float
            try:
                score = float(score_text)
                score = max(0.0, min(10.0, score))
                self.logger.info(f"Score: {score:.1f}")
                return score
            except ValueError:
                # If that fails, try to find a number with regex
                match = re.search(r"\d+\.?\d*", score_text)
                if match:
                    score = float(match.group())
                    score = max(0.0, min(10.0, score))
                    self.logger.info(f"Score: {score:.1f} (extracted from text)")
                    return score
                else:
                    self.logger.info(
                        f"Could not parse score from LLM response: '{score_text[:100]}'. "
                        f"Returning 0.0"
                    )
                    return 0.0

        except Exception as e:
            self.logger.info(f"Error scoring document: {e}")
            return 0.0


def create_gemini_reranker(config: dict[str, Any]) -> Reranker:
    """Factory function to create a Gemini reranker from config dict.

    Parameters
    ----------
    config
        Configuration dictionary with optional keys: model, api_key, max_chars, scoring_prompt

    Returns
    -------
    Configured GeminiReranker instance
    """
    return GeminiReranker(
        model=config.get("model", DEFAULT_RERANK_MODEL),
        api_key=config.get("api_key"),
        max_chars=config.get("max_chars", DEFAULT_MAX_RERANK_CHARS),
        scoring_prompt=config.get("scoring_prompt"),
    )
