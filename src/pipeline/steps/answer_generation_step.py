from __future__ import annotations

"""Step that generates the final answer from retrieved documents."""

import logging
from pathlib import Path

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
        self._logger = logging.getLogger(__name__)

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
        if not context.retrieved_docs:
            self._logger.info("No retrieved docs available. Skipping answer generation.")
            self._append_access_denial_notice(context)
            return

        context_text = self._build_context_and_citations(context)

        prompt = self.create_prompt(context_text, context.user_query)
        context.prompt = prompt

        if not self._generate_response(context, prompt):
            return

        self._append_access_denial_notice(context)
        self._logger.info("Generated final answer successfully")

    def _build_context_and_citations(self, context: QueryContext) -> str:
        """Build context text and citations from retrieved documents."""
        blocks: list[str] = []
        citations: list[dict] = []

        for i, (doc, score) in enumerate(context.retrieved_docs, start=1):
            source, page = self._extract_source_and_page(doc)
            citations.append({
                "id": i,
                "source": source,
                "page": page,
                "score": score,
                "content": doc.page_content,
            })
            blocks.append(
                f"[{i}] SOURCE={source} PAGE={page} SCORE={score}\n{doc.page_content}"
            )

        context.citations = citations

        context_text = "\n\n---\n\n".join(blocks)
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n\n[TRUNCATED]\n"

        return context_text

    def _extract_source_and_page(self, doc: Document) -> tuple[str | None, int | None]:
        """Extract source and page from document metadata."""
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or meta.get("filename")
        page = meta.get("page") or meta.get("page_number")

        # Extract only filename from path
        if source:
            source = Path(source).name

        return source, page

    def _generate_response(self, context: QueryContext, prompt: str) -> bool:
        """Generate LLM response. Returns True on success, False on failure."""
        try:
            context.llm_response = self.llm.generate(prompt)
            return True
        except Exception as e:
            context.mark_failed(f"LLM generation failed: {e}")
            return False

    def _append_access_denial_notice(self, context: QueryContext) -> None:
        """Append access denial notice if there are restricted documents."""
        if not context.has_restricted_content or not context.restricted_docs:
            return

        denial_message = self._format_access_denial_message(context.restricted_docs)
        context.llm_response = (context.llm_response or "") + denial_message
        self._logger.info(
            f"Appended access denial notice for {len(context.restricted_docs)} restricted documents"
        )

    def _format_access_denial_message(self, restricted_docs: list[dict]) -> str:
        """Format the access denial message for restricted documents."""
        seen: set[tuple[str, tuple[str, ...]]] = set()
        lines = []

        for doc in restricted_docs:
            source = doc.get("source")
            source_name = Path(source).name if source else "Unknown source"
            required_tags = tuple(sorted(doc.get("required_tags", [])))
            key = (source_name, required_tags)

            if key in seen:
                continue
            seen.add(key)

            if required_tags:
                tags_str = ", ".join(required_tags)
                lines.append(f"- {source_name} (requires: {tags_str})")
            else:
                lines.append(f"- {source_name}")

        source_list = "\n".join(lines)

        return (
            f"\n\n---\n**Note:** {len(lines)} additional document(s) "
            f"matched your query but you lack permission to access them:\n{source_list}"
        )
