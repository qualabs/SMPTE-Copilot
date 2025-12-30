#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import argparse
import json
import logging
import sys
from pathlib import Path

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


def load_role_mapping(mapping_file: str) -> dict[str, list[str]]:
    """Load role-to-tags mapping from JSON file.

    Parameters
    ----------
    mapping_file : str
        Path to the JSON file containing role-to-tags mapping.

    Returns
    -------
    dict[str, list[str]]
        Role-to-tags mapping, or empty dict if file doesn't exist.
    """
    logger = logging.getLogger()
    try:
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with open(mapping_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Role mapping file not found: {mapping_file}")
    except Exception as e:
        logger.warning(f"Could not load role mapping: {e}")
    return {}


def main():
    """Query the vector database with a question from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query vector database with optional role-aware access control"
    )
    parser.add_argument(
        "query",
        nargs="+",
        help="Search query string",
    )
    parser.add_argument(
        "--user-role",
        type=str,
        default="",
        help="User's role for access control (e.g., 'Finance_Manager')",
    )
    parser.add_argument(
        "--user-tags",
        type=str,
        default="",
        help="User's direct access tags, comma-separated (e.g., 'Finance,Public')",
    )
    parser.add_argument(
        "--role-mapping",
        type=str,
        default="",
        help="Path to role-to-tags mapping JSON file",
    )
    args = parser.parse_args()

    config = Config.get_config()

    Logger.setup(config)
    logger = logging.getLogger()

    query = " ".join(args.query)
    
    # Parse access control arguments
    user_role = args.user_role.strip() if args.user_role else None
    user_tags = [tag.strip() for tag in args.user_tags.split(",") if tag.strip()]
    role_mapping = None
    if args.role_mapping:
        role_mapping = load_role_mapping(args.role_mapping)

    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info("Querying Vector Database")
    logger.info(SEPARATOR_CHAR * SEPARATOR_LENGTH)
    logger.info(f"Query: {query}")
    if user_role:
        logger.info(f"User role: {user_role}")
    if user_tags:
        logger.info(f"User tags: {user_tags}")
    if role_mapping:
        logger.info(f"Role mapping loaded: {len(role_mapping)} roles")
    logger.info("")

    try:
        # Initialize RAG components
        components = initialize_rag_components(config)

        # Execute query using shared logic with access control
        context = execute_query(
            components,
            query,
            user_role=user_role,
            user_tags=user_tags if user_tags else None,
            role_mapping=role_mapping,
        )

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
