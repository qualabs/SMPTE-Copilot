#!/usr/bin/env python3
"""Simple script to query the vector database with a question."""

import argparse
import json
import logging
import sys
from pathlib import Path

from src import Config
from src.cli.constants import (
    EXIT_CODE_ERROR,
    SEPARATOR_CHAR,
    SEPARATOR_LENGTH,
)
from src.components import execute_query, initialize_rag_components
from src.logger import Logger
from src.pipeline import PipelineStatus, QueryContext


def _load_role_mapping(
    mapping_file: str,
    logger: logging.Logger
) -> dict[str, list[str]]:
    """Load role-to-tags mapping from JSON file.

    Parameters
    ----------
    mapping_file : str
        Path to the JSON file containing role-to-tags mapping.
    logger : logging.Logger
        Logger instance for logging messages.

    Returns
    -------
    dict[str, list[str]]
        Role-to-tags mapping, or empty dict if file doesn't exist.
    """
    try:
        mapping_path = Path(mapping_file)
        if mapping_path.exists():
            with mapping_path.open() as f:
                return json.load(f)
        else:
            logger.warning(f"Role mapping file not found: {mapping_file}")
    except Exception as e:
        logger.warning(f"Could not load role mapping: {e}")
    return {}


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
    role_mapping = {}
    if config.access_control.role_mapping_file:
        role_mapping = _load_role_mapping(str(config.access_control.role_mapping_file), logger)

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
