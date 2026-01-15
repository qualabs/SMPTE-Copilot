from __future__ import annotations

"""Utility module for initializing RAG pipeline components from configuration."""

import logging
from typing import NamedTuple

from . import (
    Config,
    EmbeddingModelFactory,
    LLMFactory,
    RerankerFactory,
    RetrieverFactory,
    VectorStoreFactory,
)
from .embeddings.protocol import Embeddings
from .llms.protocol import LLM
from .pipeline import PipelineExecutor, QueryContext
from .pipeline.steps import (
    AccessControlStep,
    GenerationStep,
    QueryEmbeddingStep,
    RerankStep,
    RetrieveStep,
)
from .rerankers.protocol import Reranker
from .retrievers.protocol import Retriever
from .vector_stores.protocol import VectorStore


class RAGComponents(NamedTuple):
    """Container for initialized RAG pipeline components."""

    embedding_model: Embeddings | None
    vector_store: VectorStore | None
    retriever: Retriever | None
    reranker: Reranker | None
    llm: LLM | None


def initialize_rag_components(config: Config | None = None) -> RAGComponents:
    """Initialize all RAG pipeline components from configuration.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, loads from Config.get_config()

    Returns
    -------
    RAGComponents
        Named tuple containing all initialized components
    """
    if config is None:
        config = Config.get_config()

    logger = logging.getLogger(__name__)
    logger.info("Initializing RAG components...")

    pipeline_config = config.pipeline.query

    embedding_model = None
    vector_store = None
    retriever = None
    reranker = None
    llm = None

    logger.info(f"Retrieve step - enabled: {pipeline_config.retrieve_enabled}")
    logger.info(f"Rerank step - enabled: {pipeline_config.rerank_enabled}")
    logger.info(f"Generation step - enabled: {pipeline_config.generation_enabled}")

    # Only create embedding, vector_store and retriever if retrieve step is enabled
    # (retrieve includes query embedding as they are dependent)
    if pipeline_config.retrieve_enabled:
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )

        store_config = {
                "embedding_function": embedding_model,
                **(config.vector_store.store_config or {}),
        }

        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            **store_config,
        )

        retriever_kwargs = {"vector_store": vector_store, "k": config.retrieval.k}
        if config.retrieval.searcher_config:
            retriever_kwargs.update(config.retrieval.searcher_config)

        retriever = RetrieverFactory.create(
            config.retrieval.searcher_strategy,
            **retriever_kwargs,
        )

    # Only create reranker if rerank step is enabled
    if pipeline_config.rerank_enabled:
        reranker = RerankerFactory.create(
            config.reranking.reranker_name,
            **(config.reranking.reranker_config or {}),
        )

    if pipeline_config.generation_enabled:
        llm = LLMFactory.create(
            config.llm.llm_name,
            **(config.llm.llm_config or {}),
        )

    logger.info("RAG components initialized successfully")

    return RAGComponents(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
    )


def execute_query(
    components: RAGComponents,
    query: str,
    user_role: str | None = None,
    role_mapping: dict[str, list[str]] | None = None,
) -> QueryContext:
    """Execute a RAG query using the provided components

    Parameters
    ----------
    components : RAGComponents
        Initialized RAG components
    query : str
        User's question or query text
    user_role : str, optional
        User's role for access control (expanded to tags via role_mapping)
    role_mapping : dict[str, list[str]], optional
        Role-to-tags mapping for access control

    Returns
    -------
    QueryContext
        Pipeline context containing the query results
    """
    config = Config.get_config()

    logger = logging.getLogger(__name__)
    logger.info(f"Executing query: {query}")

    context = QueryContext(user_query=query)

    # Set tag-based access control fields if provided
    if user_role:
        context.user_role = user_role
    if role_mapping:
        context.role_mapping = role_mapping

    # Build steps list dynamically based on pipeline configuration
    steps = []

    if components.embedding_model:
        steps.append(QueryEmbeddingStep(components.embedding_model))

    if components.retriever:
        steps.append(RetrieveStep(components.retriever))

        if config.access_control.notify_on_denied_access:
            steps.append(AccessControlStep())

    if components.reranker:
        steps.append(RerankStep(components.reranker))

    if components.llm:
        # Get prompt configuration from pipeline config
        pipeline_config = config.pipeline.query if config.pipeline else None
        prompt_template = pipeline_config.generation_prompt if pipeline_config else None

        steps.append(
            GenerationStep(
                components.llm,
                prompt_template=prompt_template,
            )
        )

    executor = PipelineExecutor(steps)
    context = executor.execute(context)

    return context
