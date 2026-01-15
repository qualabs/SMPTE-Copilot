"""Base tokenizer class that extends BaseTokenizer from docling."""

import logging
from abc import ABC, abstractmethod

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from ..constants import CHARS_PER_TOKEN_RATIO, SPLIT_BUFFER_SIZE


class Tokenizer(BaseTokenizer, ABC):
    """Base class for tokenizer implementations that extends BaseTokenizer.

    Tokenizers are used by hybrid chunkers to count tokens in text
    and split text into chunks based on token limits.
    """

    def __init__(
        self,
        chars_per_token_ratio: float | None = None,
        split_buffer_size: int | None = None,
        **kwargs,
    ):
        """Initialize the tokenizer.

        Parameters
        ----------
        chars_per_token_ratio
            Ratio of characters to tokens for threshold estimation.
            Lower values are more conservative. Default from constants.
        split_buffer_size
            Number of words to buffer before checking limits. Default from constants.
        """
        super().__init__(**kwargs)
        self.chars_per_token_ratio = chars_per_token_ratio or CHARS_PER_TOKEN_RATIO
        self.split_buffer_size = split_buffer_size or SPLIT_BUFFER_SIZE

    def __call__(self, text: str) -> int:
        """Make tokenizer callable for semchunk compatibility.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        Number of tokens in the text.
        """
        return self.count_tokens(text)

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Parameters
        ----------
        text
            The text to count tokens for.

        Returns
        -------
        Number of tokens in the text.
        """
        ...

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens allowed per chunk.

        Returns
        -------
        Maximum tokens per chunk.
        """
        ...

    def get_tokenizer(self):
        """Returns the tokenizer instance (required by BaseTokenizer).

        Returns
        -------
        The tokenizer instance (self).
        """
        return self

    def _join(self, base: str, addition: str) -> str:
        """Join two text parts with a space."""
        return f"{base} {addition}" if base else addition

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks based on token limits.

        Parameters
        ----------
        text
            Text to split.

        Returns
        -------
        List of text chunks, each within the token limit.
        """
        if not text:
            return []

        max_tokens = self.get_max_tokens()
        char_threshold = int(max_tokens * self.chars_per_token_ratio)
        buffer_size = self.split_buffer_size
        logger = logging.getLogger(__name__)
        logger.info(f"split_text called: {len(text)} chars, max_tokens={max_tokens}, char_threshold={char_threshold}")

        chunks: list[str] = []
        words = text.split()
        current_chunk = ""
        buffer: list[str] = []


        for word in words:
            buffer.append(word)
            test_chunk = self._join(current_chunk, " ".join(buffer))
            near_limit = len(test_chunk) >= char_threshold
            buffer_full = len(buffer) >= buffer_size

            if not buffer_full:
                continue

            if near_limit:
                logger.info("Near char threshold, checking token count")
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    logger.info("Exceeded token limit, starting new chunk")
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = " ".join(buffer)
            else:
                current_chunk = test_chunk

            buffer = []

        if buffer:
            current_chunk = self._join(current_chunk, " ".join(buffer))
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

