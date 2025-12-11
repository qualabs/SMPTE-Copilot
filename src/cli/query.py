#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import logging
import sys

from src import Config, EmbeddingModelFactory, RetrieverFactory, VectorStoreFactory
from src.cli.constants import (
    ALT_SEPARATOR_CHAR,
    ENUMERATE_START,
    EXIT_CODE_ERROR,
    MAX_SCORE_COSINE,
    MAX_SCORE_DISTANCE,
    MIN_SCORE,
    SCORE_DECIMAL_PLACES,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.logger import Logger


def main():
    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger()

    query = " ".join(sys.argv[1:])

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Querying Vector Database")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Query: {query}\n")

    try:
        # Step 1: Check if database exists
        vector_db_path = config.vector_store.persist_directory
        if not vector_db_path.exists():
            logger.error(f"✗ Error: Vector database not found: {vector_db_path}")
            logger.error(
                "\nPlease ingest documents first Or set RAG_CONFIG_FILE "
                "or environment variables, then run:"
            )
            logger.error(" python ingest.py")
            sys.exit(EXIT_CODE_ERROR)

        # Step 2: Create embedding model using factory
        embedding_model = EmbeddingModelFactory.create(
            config.embedding.embed_name,
            **(config.embedding.embed_config or {}),
        )

        # Step 3: Create vector store using factory
        vector_store = VectorStoreFactory.create(
            config.vector_store.store_name,
            persist_directory=str(vector_db_path),
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_model,
            **(config.vector_store.store_config or {}),
        )

        # Step 4: Create retriever
        searcher_config = {"k": config.retrieval.k}
        if config.retrieval.searcher_config:
            searcher_config.update(config.retrieval.searcher_config)
        searcher_config["vector_store"] = vector_store

        retriever = RetrieverFactory.create(
            config.retrieval.searcher_strategy,
            **searcher_config,
        )

        # Step 5: Query the database using retriever (with scores)
        logger.info("Searching...")
        results_with_scores = retriever.retrieve_with_scores(query)

        # Step 6: Display results with similarity scores
        logger.info(f"\nFound {len(results_with_scores)} relevant documents:\n")
        logger.info(ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Similarity Score Guide:")
        logger.info("  - Higher score = More similar to query")
        logger.info(
            f"  - Score range depends on distance metric "
            f"(usually {MIN_SCORE}-{MAX_SCORE_COSINE} or "
            f"{MIN_SCORE}-{MAX_SCORE_DISTANCE})"
        )
        logger.info(f"  - For cosine similarity: closer to {MAX_SCORE_COSINE} = more similar")
        logger.info(ALT_SEPARATOR_CHAR * SEPARATOR_LENGTH)

        for i, (doc, score) in enumerate(results_with_scores, ENUMERATE_START):
            logger.info(f"\n[{i}] Similarity Score: {score:.{SCORE_DECIMAL_PLACES}f}")
            logger.info(f"    Content: {doc.page_content}")
            if doc.metadata:
                logger.info(f"    Metadata: {doc.metadata}")

        logger.info("\n" + SEPARATOR_CHAR * SEPARATOR_LENGTH)
        logger.info("Note: Higher similarity scores indicate better matches to your query.")
        logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)

    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)

if __name__ == "__main__":
    main()
