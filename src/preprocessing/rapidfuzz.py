from __future__ import annotations

"""RapidFuzz-based preprocessor implementation."""

import re
from typing import Any

import logging
from rapidfuzz import fuzz

from .protocol import Preprocessor

logger = logging.getLogger(__name__)


class RapidFuzzPreprocessor:
    """Preprocessor using RapidFuzz for fuzzy matching to remove repeated content.

    This is a concrete implementation of the Preprocessor protocol using
    RapidFuzz to detect and remove similar lines that appear multiple times,
    such as headers, footers, and page numbers.
    """

    def __init__(
        self,
        min_repetitions: int = 3,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the preprocessor.

        Parameters
        ----------
        min_repetitions
            Minimum number of times a line must appear (or be similar) to be
            considered repeated. Default is 3. This prevents removing legitimate
            content that appears a few times.
        similarity_threshold
            Minimum similarity ratio (0.0 to 1.0) for considering lines as similar.
            Uses rapidfuzz for fuzzy matching. Default is 0.85 (85% similarity).
        """
        self.min_repetitions = min_repetitions
        self.similarity_threshold = similarity_threshold

    def preprocess(self, text: str) -> str:
        """Preprocess text to remove repeated content.

        Parameters
        ----------
        text
            The text to preprocess.

        Returns
        -------
        Preprocessed text with repeated content removed.
        """
        if not text or not text.strip():
            return text

        lines = text.split("\n")
        if len(lines) < self.min_repetitions:
            return text

        lines_to_remove = self._detect_repeated_lines(lines)

        cleaned_lines = [
            line for i, line in enumerate(lines) if i not in lines_to_remove
        ]

        removed_count = len(lines_to_remove)
        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} lines of repeated content "
                f"({removed_count / len(lines) * 100:.1f}% of total lines)"
            )

        return "\n".join(cleaned_lines)

    def _detect_repeated_lines(self, lines: list[str]) -> set[int]:
        """Detect lines that appear multiple times using fuzzy matching.

        Parameters
        ----------
        lines
            List of text lines.

        Returns
        -------
        Set of line indices to remove.
        """
        normalized_lines = [
            self._normalize_line(line) if line.strip() else None
            for line in lines
        ]

        if len([n for n in normalized_lines if n is not None]) < self.min_repetitions:
            return set()

        indices_to_remove = set()
        processed = set()

        for i, line in enumerate(normalized_lines):
            if line is None or i in processed:
                continue

            similar_indices = [i]
            
            for j in range(i + 1, len(normalized_lines)):
                if normalized_lines[j] is None or j in processed:
                    continue
                    
                similarity = fuzz.ratio(
                    line, normalized_lines[j],
                    score_cutoff=int(self.similarity_threshold * 100)
                )
                if similarity >= self.similarity_threshold * 100:
                    similar_indices.append(j)
                    processed.add(j)

            if len(similar_indices) >= self.min_repetitions:
                indices_to_remove.update(similar_indices)
            
            processed.add(i)

        return indices_to_remove

    @staticmethod
    def _normalize_line(line: str) -> str:
        """Normalize a line for comparison.

        Parameters
        ----------
        line
            The line to normalize.

        Returns
        -------
        Normalized line string.
        """
        normalized = line.strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized


def create_rapidfuzz_preprocessor(config: dict[str, Any]) -> Preprocessor:
    """Create a RapidFuzz preprocessor from configuration.

    Parameters
    ----------
    config
        Configuration dictionary. Expected keys:
        - min_repetitions (int, optional): Minimum repetitions. Default: 3
        - similarity_threshold (float, optional): Similarity threshold. Default: 0.85

    Returns
    -------
    Preprocessor instance.
    """
    min_repetitions = config.get("min_repetitions", 3)
    similarity_threshold = config.get("similarity_threshold", 0.85)
    
    return RapidFuzzPreprocessor(
        min_repetitions=min_repetitions,
        similarity_threshold=similarity_threshold,
    )

