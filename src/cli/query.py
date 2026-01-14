#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import argparse
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
from src.pipeline import PipelineStatus, QueryContext


def _log_query_info(
    query: str,
    user_role: str | None,
    role_mapping: dict[str, list[str]],
    logger: logging.Logger
) -> None:
    """Log query information and role mapping."""
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Querying Vector Database")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Query: {query}")
    if user_role:
        logger.info(f"User role: {user_role}")
    if role_mapping:
        logger.info(f"Role mapping loaded: {len(role_mapping)} roles")
        if user_role and user_role in role_mapping:
            logger.info(f"Role '{user_role}' maps to tags: {role_mapping[user_role]}")
    logger.info("")


def _display_results(
    context: QueryContext,
    logger: logging.Logger,
) -> None:
    """Display query results, citations, and retrieved documents."""
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


def main():
    """Query the vector database with a question from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query vector database with role-aware access control from config"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query string",
    )
    args = parser.parse_args()

    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger(__name__)

    query = " ".join(args.query)

    user_role = config.access_control.default_user_role
    role_mapping = config.access_control.get_role_mapping()

    _log_query_info(query, user_role, role_mapping, logger)

    try:
        components = initialize_rag_components(config)

        context = execute_query(
            components,
            query,
            user_role=user_role,
            role_mapping=role_mapping,
        )

        if context.status == PipelineStatus.FAILED:
            raise RuntimeError(f"Pipeline failed: {context.error}")

        _display_results(context, logger)

    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(EXIT_CODE_ERROR)

if __name__ == "__main__":
    main()
