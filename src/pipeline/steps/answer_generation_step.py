from __future__ import annotations

"""Step that generates the final answer from retrieved documents."""

import logging

from langchain_core.documents import Document

from ...llms.protocol import LLM
from ..contexts.query_context import QueryContext
from ..step import PipelineStep


class GenerationStep(PipelineStep):
    """Step that generates the final answer from retrieved documents."""

    def __init__(self, llm: LLM,  max_context_chars: int = 12000):
        """Initialize the generation step.

        Parameters
        ----------
        llm
            LLM instance created by LLMFactory (or wired manually).
        max_context_chars
            Max characters of retrieved context injected into the prompt.
        """
        self.llm = llm
        self.max_context_chars = max_context_chars

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

        prompt = f"""You are SMPTE-Copilot, an expert technical assistant.

Your task is to answer the user's question based on the provided context documents.

Guidelines:
- Synthesize and integrate information from the context to provide a comprehensive answer
- Be concise but thorough, using technical terminology when appropriate
- Always cite sources using [1], [2], etc., referring to the context blocks
- If the context contains relevant information but doesn't directly answer the question, explain what the context reveals about the topic
- Only say "I don't know based on the provided documents" if the context is completely unrelated to the question
- Do not fabricate information that isn't supported by the context

Context Documents:
{context_text}

Question:
{context.user_query}

Answer:
""".strip()

        context.prompt = prompt
        context.citations = citations

        try:
            context.llm_response = self.llm.generate(prompt)
        except Exception as e:
            context.mark_failed(f"LLM generation failed: {e}")
            return

        logger.info("Generated final answer successfully")
