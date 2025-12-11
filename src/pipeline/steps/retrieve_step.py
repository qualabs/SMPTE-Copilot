"""Step that retrieves relevant documents from the vector store."""
from __future__ import annotations

import logging

from ...retrievers.protocol import Retriever
from ..contexts.query_context import QueryContext
from ..step import PipelineStep


class RetrieveStep:
    """Step that retrieves relevant documents from the vector store."""

    def __init__(self, retriever: Retriever):
        """Initialize the retrieve step.

        Parameters
        ----------
        retriever
            Retriever instance created by RetrieverFactory.
        """
        self.retriever = retriever

    def run(self, context: QueryContext) -> None:
        """Retrieve relevant documents for the query.

        Parameters
        ----------
        context
            Query context with user_query set.
        """
        logger = logging.getLogger()
        logger.info(f"Retrieving documents for query: {context.user_query}")

        results_with_scores = self.retriever.retrieve_with_scores(context.user_query)
        context.retrieved_docs = results_with_scores

        logger.info(f"Retrieved {len(results_with_scores)} documents")
