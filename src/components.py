"""Utility module for initializing RAG pipeline components from configuration."""
from __future__ import annotations

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
from .retrievers.protocol import Retriever
from .vector_stores.protocol import VectorStore


class RAGComponents(NamedTuple):
    """Container for initialized RAG pipeline components."""

    embedding_model: Embeddings
    vector_store: VectorStore
    retriever: Retriever
    llm: LLM
    config: Config


def initialize_rag_components(config: Config | None = None) -> RAGComponents:
    """Initialize all RAG pipeline components from configuration.

    This function creates and wires together all the components needed for
    the RAG pipeline: embedding model, vector store, retriever, and LLM.

    Parameters
    ----------
    config : Config, optional
        Configuration object. If None, loads from Config.get_config()

    Returns
    -------
    RAGComponents
        Named tuple containing all initialized components

    Raises
    ------
    RuntimeError
        If vector database doesn't exist or initialization fails
    """
    if config is None:
        config = Config.get_config()

    logger = logging.getLogger()

    # Check if vector database exists
    vector_db_path: Path = config.vector_store.persist_directory
    if not vector_db_path.exists():
        raise RuntimeError(
            f"Vector database not found at {vector_db_path}. "
            "Please run ingestion first."
        )

    logger.info("Initializing RAG components...")

    # Initialize embedding model
    embedding_model = EmbeddingModelFactory.create(
        config.embedding.embed_name,
        **(config.embedding.embed_config or {}),
    )

    # Initialize vector store
    vector_store = VectorStoreFactory.create(
        config.vector_store.store_name,
        persist_directory=str(vector_db_path),
        collection_name=config.vector_store.collection_name,
        embedding_function=embedding_model,
        **(config.vector_store.store_config or {}),
    )

    # Initialize retriever
    searcher_config = {"k": config.retrieval.k}
    if config.retrieval.searcher_config:
        searcher_config.update(config.retrieval.searcher_config)
    searcher_config["vector_store"] = vector_store

    retriever = RetrieverFactory.create(
        config.retrieval.searcher_strategy,
        **searcher_config,
    )

    # Initialize LLM
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
        config=config,
    )
