"""Shared constants used across multiple modules.

This module contains only constants that are truly shared across different
domains/modules. Domain-specific constants should be in their respective
modules (e.g., chunkers/constants.py, cli/constants.py, etc.).
"""

# ============================================================================
# Encoding (used across multiple modules: chunkers, loaders, protocols)
# ============================================================================
DEFAULT_ENCODING = "utf-8"

# ============================================================================
# Retrieval (used across retrievers, vector_stores, and config)
# ============================================================================
DEFAULT_RETRIEVAL_K = 5

