"""Constants for reranking module."""

DEFAULT_RERANK_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_RERANK_CHARS = 2000

DEFAULT_SCORING_PROMPT = """Rate document relevance from 0-10.

Query: {query}

Document:
{document}

Instructions: Return ONLY a single number from 0 to 10.
- 10 = Directly answers with details
- 5 = Partially relevant
- 0 = Not relevant

Output format: Just the number, nothing else.
"""
