from __future__ import annotations

"""RapidFuzz-based preprocessor implementation."""

import re
from typing import Any
from collections import defaultdict

import logging
from rapidfuzz import fuzz

from .protocol import Preprocessor

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
        self.logger = logging.getLogger(__name__)
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

        removed_count = len(lines_to_remove)
        if removed_count > 0:
            self.logger.info(
                f"Removed {removed_count} lines of repeated content "
                f"({removed_count / len(lines) * 100:.1f}% of total lines)"
            )
            for line_idx in sorted(lines_to_remove):
                line_content = lines[line_idx].strip()[:100]  # Limit to 100 chars for readability
                self.logger.info(f"Deleted line {line_idx + 1}: {line_content}")

        cleaned_lines = [
            line for i, line in enumerate(lines) if i not in lines_to_remove
        ]

        return "\n".join(cleaned_lines)

    def _detect_repeated_lines(self, lines: list[str]) -> set[int]:
        """Detect lines that appear multiple times using fuzzy matching.

        Uses a two-phase optimization:
        1. Pre-filter by line characteristics (length, prefix) to group candidates
        2. Fuzzy matching only within candidate groups

        This reduces complexity from O(n²) to O(n*k) where k is typically much
        smaller than n, even in worst-case scenarios.

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

        non_empty_count = sum(1 for n in normalized_lines if n is not None)
        if non_empty_count < self.min_repetitions:
            return set()

        candidate_groups = self._group_by_characteristics(normalized_lines)
        indices_to_remove = set()

        numeric_lines_indices = self._group_numeric_lines(normalized_lines)
        if len(numeric_lines_indices) >= self.min_repetitions:
            indices_to_remove.update(numeric_lines_indices)

        for candidate_group_indices in candidate_groups.values():
            if len(candidate_group_indices) < self.min_repetitions:
                continue

            clusters: list[tuple[str, list[int]]] = []
            for i in candidate_group_indices:
                if i in indices_to_remove:
                    continue
                    
                line = normalized_lines[i]
                matched_cluster_idx = None
                for cluster_idx, cluster in enumerate(clusters):
                    representative = cluster[0]
                    similarity = fuzz.ratio(
                        line, representative,
                        score_cutoff=int(self.similarity_threshold * 100)
                    )
                    if similarity >= self.similarity_threshold * 100:
                        matched_cluster_idx = cluster_idx
                        break

                if matched_cluster_idx is not None:
                    clusters[matched_cluster_idx][1].append(i)
                else:
                    clusters.append((line, [i]))

            for cluster in clusters:
                indices = cluster[1]
                if len(indices) >= self.min_repetitions:
                    indices_to_remove.update(indices)

        return indices_to_remove

    def _group_numeric_lines(self, normalized_lines: list[str | None]) -> set[int]:
        """Group lines that contain only numbers.

        This handles cases like page numbers "1", "2", "3" that should be
        removed as repetitive content even though they have low similarity.

        Parameters
        ----------
        normalized_lines
            List of normalized lines (may contain None).

        Returns
        -------
        Set of line indices that are numeric-only and should be removed.
        """
        numeric_indices = []
        for i, line in enumerate(normalized_lines):
            if line is None:
                continue
            if re.match(r"^\d+$", line.strip()):
                numeric_indices.append(i)

        if len(numeric_indices) >= self.min_repetitions:
            return set(numeric_indices)
        return set()

    def _group_by_characteristics(self, normalized_lines: list[str | None]) -> dict[tuple, list[int]]:
        """Group lines by characteristics to reduce fuzzy matching comparisons.

        Groups lines by (length_category, prefix) where:
        - length_category: rounded length to reduce groups
        - prefix: first few characters for quick filtering

        Only stores indices to minimize memory usage. Lines are accessed from
        normalized_lines when needed.

        Parameters
        ----------
        normalized_lines
            List of normalized lines (may contain None).

        Returns
        -------
        Dictionary mapping (length_category, prefix) to list of line indices.
        """
        groups = defaultdict(list)
        prefix_len = 10

        for i, line in enumerate(normalized_lines):
            if line is None:
                continue

            length = len(line)
            length_category = length // 20
            prefix = line[:prefix_len] if length >= prefix_len else line

            groups[(length_category, prefix)].append(i)

        return groups

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

