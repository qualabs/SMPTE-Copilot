from __future__ import annotations

"""Step that generates the final answer from retrieved documents."""

import logging

from langchain_core.documents import Document

from ...llms.protocol import LLM
from ..contexts.query_context import QueryContext
from ..step import PipelineStep
from .constants import DEFAULT_GENERATION_PROMPT, DEFAULT_MAX_CONTEXT_CHARS


class GenerationStep(PipelineStep):
    """Step that generates the final answer from retrieved documents."""

    def __init__(
        self,
        llm: LLM,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        prompt_template: str | None = None,
    ):
        """Initialize the generation step.

        Parameters
        ----------
        llm
            LLM instance created by LLMFactory (or wired manually).
        max_context_chars
            Max characters of retrieved context injected into the prompt.
        prompt_template
            Custom prompt template (use {context} and {query} placeholders).
            If None, uses default prompt.
        """
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.prompt_template = prompt_template or DEFAULT_GENERATION_PROMPT

    def create_prompt(self, context_text: str, query: str) -> str:
        """Create the generation prompt using template.

        Parameters
        ----------
        context_text
            Formatted context from retrieved documents
        query
            User's query

        Returns
        -------
        Formatted prompt string for LLM generation
        """
        return self.prompt_template.format(
            context=context_text,
            query=query
        ).strip()

    def run(self, context: QueryContext) -> None:
        """Generate the final answer.

        Parameters
        ----------
        context
            Query context with retrieved_docs set.
        """
        logger = logging.getLogger(__name__)

        if not context.retrieved_docs:
            logger.info("No retrieved docs available. Skipping answer generation.")
            return

        retrieved: list[tuple[Document, float]] = context.retrieved_docs

        blocks: list[str] = []
        citations: list[dict] = []

        for i, (doc, score) in enumerate(retrieved, start=1):
            meta = doc.metadata or {}
            source = meta.get("source") or meta.get("file_path") or meta.get("filename")
            page = meta.get("page") or meta.get("page_number")

            citations.append(
                {"id": i, "source": source, "page": page, "score": score}
            )

            blocks.append(
                f"[{i}] SOURCE={source} PAGE={page} SCORE={score}\n{doc.page_content}"
            )

        context_text = "\n\n---\n\n".join(blocks)
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n\n[TRUNCATED]\n"

        prompt = self.create_prompt(context_text, context.user_query)
        context.prompt = prompt
        context.citations = citations

        try:
            context.llm_response = self.llm.generate(prompt)
        except Exception as e:
            context.mark_failed(f"LLM generation failed: {e}")
            return

        logger.info("Generated final answer successfully")
