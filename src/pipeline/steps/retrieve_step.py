from __future__ import annotations

"""Step that retrieves relevant documents from the vector store."""

import logging

from ...config import Config
from ...retrievers.filters import FilterBuilderFactory
from ...retrievers.protocol import Retriever
from ..contexts.query_context import QueryContext


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

        # Build access filter if tag-based access control is enabled
        # Roles are automatically converted to tags via role_mapping
        metadata_filter = None
        if context.user_role and context.role_mapping:
            logger.info(
                f"Applying tag-based access control: role='{context.user_role}'"
            )

            # Get vector store type from configuration
            config = Config.get_config()
            vector_store_type = config.vector_store.store_name

            try:
                builder = FilterBuilderFactory.create(vector_store_type)
                metadata_filter = builder.build(
                    user_role=context.user_role,
                    role_mapping=context.role_mapping,
                )
                self.retriever.set_filter(metadata_filter)
                logger.debug("Set metadata filter on retriever")
            except ValueError as e:
                logger.warning(f"Could not create filter builder: {e}")
                metadata_filter = None

        logger.info(f"Retrieving documents for query: {context.user_query}")

        results_with_scores = self.retriever.retrieve_with_scores(context.user_query)
        context.retrieved_docs = results_with_scores

        logger.info(f"Retrieved {len(results_with_scores)} documents")
