from __future__ import annotations

"""Step that retrieves relevant documents from the vector store."""

import logging

from ...retrievers.filters import build_access_filter
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
        logger = logging.getLogger(__name__)
        
        # Build access filter if role-aware access control is enabled
        if context.user_role or context.user_tags:
            logger.info(
                f"Applying role-aware access control: role='{context.user_role}', "
                f"tags={context.user_tags}"
            )
            context.metadata_filter = build_access_filter(
                user_role=context.user_role,
                user_tags=context.user_tags,
                role_mapping=context.role_mapping,
            )
        
        # Apply metadata filter to retriever if provided
        if context.metadata_filter is not None and hasattr(self.retriever, "metadata_filter"):
            self.retriever.metadata_filter = context.metadata_filter
            logger.debug("Applied metadata filter to retriever")
        
        
        logger.info(f"Retrieving documents for query: {context.user_query}")

        results_with_scores = self.retriever.retrieve_with_scores(context.user_query)
        context.retrieved_docs = results_with_scores

        logger.info(f"Retrieved {len(results_with_scores)} documents")
