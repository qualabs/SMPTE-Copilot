#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import logging
import sys

from src import (
    Config,
    EmbeddingModelFactory,
    RetrieverFactory,
    VectorStoreFactory,
    LLMFactory,
)
from src.cli.constants import (
    ALT_SEPARATOR_CHAR,
    ENUMERATE_START,
    EXIT_CODE_ERROR,
    MAX_SCORE_DISTANCE,
    MIN_SCORE,
    SCORE_DECIMAL_PLACES,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.logger import Logger
from src.pipeline import PipelineExecutor, PipelineStatus, QueryContext
from src.pipeline.steps import QueryEmbeddingStep, RetrieveStep, GenerationStep


def main():
    """Query the vector database with a question from command line arguments."""
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger()

    query = " ".join(sys.argv[1:])

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Querying Vector Database")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Query: {query}\n")

    try:
        vector_db_path = config.vector_store.persist_directory
        if not vector_db_path.exists():
            logger.error(f"✗ Error: Vector database not found: {vector_db_path}")
            logger.error(
                "\nPlease ingest documents first Or set RAG_CONFIG_FILE "
                "or environment variables, then run:"
            )
            logger.error(" python ingest.py")
            sys.exit(EXIT_CODE_ERROR)

        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )

        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            persist_directory=str(vector_db_path),
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_model,
            **(config.vector_store.store_config or {}),
        )

        searcher_config = {"k": config.retrieval.k}
        if config.retrieval.searcher_config:
            searcher_config.update(config.retrieval.searcher_config)
        searcher_config["vector_store"] = vector_store

        retriever = RetrieverFactory.create(
            config.retrieval.searcher_strategy,
            **searcher_config,
        )

        llm = LLMFactory.create(
            config.llm.llm_name,
            **(config.llm.llm_config or {}),
        )

        context = QueryContext(user_query=query)

        steps = [
            QueryEmbeddingStep(embedding_model),
            RetrieveStep(retriever),
            GenerationStep(llm, config.llm.llm_name),
        ]

        executor = PipelineExecutor(steps)
        context = executor.execute(context)

        if context.status == PipelineStatus.FAILED:
            raise RuntimeError(f"Pipeline failed: {context.error}")

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Final Answer")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info(context.llm_response or "(no response)")

        if context.citations:
            logger.info("\nSources:")
            for c in context.citations:
                cid = c.get("id")
                source = c.get("source")
                score = c.get("score")
                logger.info(f"  [{cid}] {source}  distance={score}")

        results_with_scores = context.retrieved_docs or []

        logger.info("\n" + ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info(f"Retrieved {len(results_with_scores)} chunks (debug):")
        logger.info("Distance Score Guide (similarity_search_with_score):")
        logger.info("  - Lower score = More similar to query")
        logger.info(
            f"  - Score range depends on the distance metric "
            f"(often around {MIN_SCORE}-{MAX_SCORE_DISTANCE})"
        )
        logger.info("  - Closer to 0 = more similar")
        logger.info(ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)

        for i, (doc, score) in enumerate(results_with_scores, ENUMERATE_START):
            logger.info(f"\n[{i}] Distance Score: {score:.{SCORE_DECIMAL_PLACES}f}")
            logger.info(f"    Content: {doc.page_content}")
            if doc.metadata:
                logger.info(f"    Metadata: {doc.metadata}")

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Note: Lower distance scores indicate better matches to your query.")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()
