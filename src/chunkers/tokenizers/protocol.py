"""Protocol for tokenizer implementations."""

from typing import Protocol


class Tokenizer(Protocol):
    """Protocol that all tokenizer implementations must follow.
    
    Tokenizers are used by hybrid chunkers to count tokens in text
    and split text into chunks based on token limits.
    """

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

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens allowed per chunk.

        Returns
        -------
        Maximum tokens per chunk.
        """
        ...

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

        chunks = []
        words = text.split()
        current_chunk = []

        for word in words:
            test_chunk = " ".join(current_chunk + [word])
            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk.append(word)
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

