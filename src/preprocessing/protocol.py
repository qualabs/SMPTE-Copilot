from __future__ import annotations

"""Protocol for preprocessor implementations."""

from typing import Protocol


class Preprocessor(Protocol):
    """Protocol for preprocessor implementations.

    Any class implementing this method can preprocess documents to remove
    repeated content, clean text, etc. This allows swapping preprocessing
    strategies without changing the rest of the code.
    """

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
        ...

