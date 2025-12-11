"""Step that generates an embedding for the user query."""
from __future__ import annotations

import logging

from ...embeddings.protocol import Embeddings
from ..contexts.query_context import QueryContext
from ..step import PipelineStep


class QueryEmbeddingStep:
    """Step that generates an embedding for the user query."""

    def __init__(self, embedding_model: Embeddings):
        """Initialize the query embedding step.

        Parameters
        ----------
        embedding_model
            Embedding model instance created by EmbeddingModelFactory.
        """
        self.embedding_model = embedding_model

    def run(self, context: QueryContext) -> None:
        """Generate embedding for the user query.

        Parameters
        ----------
        context
            Query context with user_query set.
        """
        logger = logging.getLogger()
        logger.info(f"Generating embedding for query: {context.user_query}")

        query_vector = self.embedding_model.embed_query(context.user_query)
        context.query_vector = query_vector

        logger.info("Query embedding generated")
