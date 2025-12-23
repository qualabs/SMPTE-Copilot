"""Constants specific to chunking functionality."""

CHUNKING_METHOD_RECURSIVE = "recursive"
CHUNKING_METHOD_CHARACTER = "character"
CHUNKING_METHOD_TOKEN = "token"

RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

CHUNK_INDEX_METADATA_KEY = "chunk_index"
TOTAL_CHUNKS_METADATA_KEY = "total_chunks"

