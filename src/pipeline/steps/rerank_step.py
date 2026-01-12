from __future__ import annotations

"""Step that reranks retrieved documents using a cross-encoder."""

import logging

from ...rerankers.protocol import Reranker
from ..contexts.query_context import QueryContext
from ..step import PipelineStep


class RerankStep(PipelineStep):
    """Step that reranks retrieved documents using a cross-encoder.

    This step takes documents from initial retrieval (with vector similarity scores)
    and reorders them using a more sophisticated cross-encoder model that can
    better assess relevance between the query and document content.
    """

    def __init__(self, reranker: Reranker):
        """Initialize the rerank step.

        Parameters
        ----------
        reranker
            Reranker instance created by RerankerFactory.
        """
        self.reranker = reranker

    def run(self, context: QueryContext) -> None:
        """Rerank retrieved documents for improved relevance.

        Parameters
        ----------
        context
            Query context with retrieved_docs set from RetrieveStep.
        """
        logger = logging.getLogger(__name__)

        if not context.retrieved_docs:
            logger.info("No documents to rerank. Skipping rerank step.")
            return

        initial_count = len(context.retrieved_docs)
        logger.info(f"Reranking {initial_count} retrieved documents")

        reranked_docs = self.reranker.rerank(
            context.user_query,
            context.retrieved_docs
        )

        context.retrieved_docs = reranked_docs

        logger.info(
            "Reranking completed. Document order may have changed based on relevance."
        )

        for i, (doc, score) in enumerate(reranked_docs[:3], 1):
            source = doc.metadata.get("source", "unknown")
            logger.debug(f"  [{i}] score={score:.2f} source={source}")
