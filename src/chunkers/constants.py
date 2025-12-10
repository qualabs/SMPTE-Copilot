"""Constants specific to chunking functionality."""

# Chunking methods
CHUNKING_METHOD_RECURSIVE = "recursive"
CHUNKING_METHOD_CHARACTER = "character"
CHUNKING_METHOD_TOKEN = "token"

# Recursive chunking separators (in order of priority)
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Default chunking values (fallback - should ideally come from config)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Metadata keys for chunk information
CHUNK_INDEX_METADATA_KEY = "chunk_index"
TOTAL_CHUNKS_METADATA_KEY = "total_chunks"

