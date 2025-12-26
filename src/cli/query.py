#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import logging
import sys

from src import Config
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
from src.components import execute_query, initialize_rag_components
from src.logger import Logger
from src.pipeline import PipelineStatus


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
        # Initialize RAG components
        components = initialize_rag_components(config)

        # Execute query using shared logic
        context = execute_query(components, query)

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
