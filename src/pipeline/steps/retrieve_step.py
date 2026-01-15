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
        self.config = Config.get_config()
        self._logger = logging.getLogger(__name__)

    def run(self, context: QueryContext) -> None:
        """Retrieve relevant documents for the query.

        Parameters
        ----------
        context
            Query context with user_query set.
        """

        if self._should_apply_db_filter(context):
            self._logger.info(
                f"Applying tag-based access control (silent mode): role='{context.user_role}'"
            )
            self._build_and_set_filter(context)

        self._retrieve_documents(context)

    def _should_apply_db_filter(self, context: QueryContext) -> bool:
        """Determine if DB-level filtering should be applied.

        Returns False if:
        - No access control is configured
        - notify_on_denied_access is enabled (filtering happens in AccessControlStep)
        """
        if not context.user_role or not context.role_mapping:
            return False

        if self.config.access_control.notify_on_denied_access:
            self._logger.info(
                "Skipping DB filter - notify_on_denied_access enabled, AccessControlStep will handle filtering"
            )
            return False

        return True

    def _build_and_set_filter(self, context: QueryContext) -> None:
        """Build and set the metadata filter on the retriever."""
        vector_store_type = self.config.vector_store.store_name

        try:
            builder = FilterBuilderFactory.create(vector_store_type)
            metadata_filter = builder.build(
                user_role=context.user_role,
                role_mapping=context.role_mapping,
            )
            self.retriever.set_filter(metadata_filter)
            self._logger.info(f"Set metadata filter on retriever: {metadata_filter}")
        except ValueError as e:
            self._logger.warning(f"Could not create filter builder: {e}")

    def _retrieve_documents(self, context: QueryContext) -> None:
        """Execute the retrieval and store results in context."""
        self._logger.info(f"Retrieving documents for query: {context.user_query}")

        results_with_scores = self.retriever.retrieve_with_scores(context.user_query)
        context.retrieved_docs = results_with_scores

        self._logger.info(f"Retrieved {len(results_with_scores)} documents")
