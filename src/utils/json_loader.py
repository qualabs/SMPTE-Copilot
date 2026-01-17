"""JSON file loading utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json_file(file_path: Path | str | None) -> dict[str, Any] | None:
    """Load a JSON file safely with proper error handling.

    Parameters
    ----------
    file_path
        Path to the JSON file. Can be None, in which case None is returned.

    Returns
    -------
    dict[str, Any] | None
        Parsed JSON data as a dictionary, or None if:
        - file_path is None
        - File doesn't exist
        - File can't be parsed as JSON
        - JSON root is not a dictionary
    """
    if file_path is None:
        return None

    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            logger.warning(f"JSON file not found: {path}")
            return None

        with path.open() as f:
            data = json.load(f)

        if not isinstance(data, dict):
            logger.error(f"Invalid JSON format: expected dict, got {type(data).__name__}")
            return None

        return data

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file '{file_path}': {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to load JSON file '{file_path}': {e}") from e
