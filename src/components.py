from __future__ import annotations

"""Utility module for initializing RAG pipeline components from configuration."""

import logging
from pathlib import Path
from typing import NamedTuple

from . import (
    Config,
    EmbeddingModelFactory,
    LLMFactory,
    RetrieverFactory,
    VectorStoreFactory,
)
from .embeddings.protocol import Embeddings
from .llms.protocol import LLM
from .pipeline import PipelineExecutor, QueryContext
from .pipeline.steps import GenerationStep, QueryEmbeddingStep, RetrieveStep
from .retrievers.protocol import Retriever
from .vector_stores.protocol import VectorStore


class RAGComponents(NamedTuple):
    """Container for initialized RAG pipeline components."""

    embedding_model: Embeddings
    vector_store: VectorStore
    retriever: Retriever
    llm: LLM


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

    if not vector_db_path.exists():
        raise RuntimeError(
            f"Vector database not found at {vector_db_path}. "
            "Please run ingestion first."
        )

    logger.info("Initializing RAG components...")

    embedding_model = EmbeddingModelFactory.create(
        config.embedding.embed_name,
        **(config.embedding.embed_config or {}),
    )

    store_config = {
        "persist_directory": config.vector_store.store_config.get("persist_directory"),
        "collection_name": config.vector_store.store_config.get("collection_name"),
        "embedding_function": embedding_model,
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

    llm = LLMFactory.create(
        config.llm.llm_name,
        **(config.llm.llm_config or {}),
    )

    logger.info("RAG components initialized successfully")

    return RAGComponents(
        embedding_model=embedding_model,
        vector_store=vector_store,
        retriever=retriever,
        llm=llm,
    )


def execute_query(
    components: RAGComponents,
    query: str,
    user_role: str = None,
    user_tags: list[str] = None,
    role_mapping: dict[str, list[str]] = None,
) -> QueryContext:
    """Execute a RAG query using the provided components

    Parameters
    ----------
    components : RAGComponents
        Initialized RAG components
    query : str
        User's question or query text
    user_role : str, optional
        User's role for access control
    user_tags : list[str], optional
        User's access tags for access control
    role_mapping : dict[str, list[str]], optional
        Role-to-tags mapping for access control

    Returns
    -------
    QueryContext
        Pipeline context containing the query results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Executing query: {query}")

    context = QueryContext(user_query=query)
    
    # Set role-aware access control fields if provided
    if user_role:
        context.user_role = user_role
    if user_tags:
        context.user_tags = user_tags
    if role_mapping:
        context.role_mapping = role_mapping

    steps = [
        QueryEmbeddingStep(components.embedding_model),
        RetrieveStep(components.retriever),
        GenerationStep(components.llm),
    ]

    executor = PipelineExecutor(steps)
    context = executor.execute(context)

    return context
