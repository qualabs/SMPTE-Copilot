# SMPTE-Copilot
An open-source AI co-pilot that ingests and indexes text, audio, and video to enable semantic, multimodal search of media archives. The prototype provides modular ingestion, a chat-based retrieval pipeline, transparent citations, and tiered access for public users, members, and staff.

## Project Structure

The project is organized into modular components that follow a consistent pattern. Each module implements the Factory pattern to enable easy extension and addition of new components.

```
SMPTE-Copilot/
├── src/
│   ├── chunkers/          # Module for splitting documents into chunks
│   ├── embeddings/        # Module for embedding models
│   ├── loaders/           # Module for loading documents from various sources
│   ├── retrievers/        # Module for document retrieval
│   ├── vector_stores/     # Module for vector storage
│   ├── config/            # Project configuration
│   └── cli/               # Command-line interfaces
├── data/                  # Data and documents to process
├── config.yaml           # Main configuration file
└── docker-compose.yml    # Docker configuration
```

### Module Architecture

All main modules (`chunkers`, `embeddings`, `loaders`, `retrievers`, `vector_stores`) follow the same architectural structure based on the Factory pattern. This consistency facilitates code understanding and the incorporation of new components.

#### Module Structure (Example: `embeddings/`)

```
embeddings/
├── __init__.py           # Exports main classes and types
├── protocol.py           # Defines the Protocol interface that all components must implement
├── types.py              # Defines the Enum with available types
├── factory.py            # Implements the Factory pattern with component registration
├── constants.py          # Module-specific constants (optional)
├── helpers.py            # Helper functions (optional)
├── huggingface.py        # Specific implementation: HuggingFace embeddings
└── openai.py             # Specific implementation: OpenAI embeddings
```

**Main components:**

1. **`protocol.py`**: Defines a Protocol (interface) that specifies the methods all implementations must provide. This ensures compatibility and allows swapping implementations without changing the rest of the code.

2. **`types.py`**: Contains an Enum that lists all available component types in the module (e.g., `EmbeddingModelType.HUGGINGFACE`, `EmbeddingModelType.OPENAI`).

3. **`factory.py`**: Implements the Factory pattern with dynamic registration. Allows registering new implementations and creating them by type. The factory maintains a dictionary that maps types to creation functions.

4. **Implementation files** (e.g., `huggingface.py`, `openai.py`): Each file contains a `create_*` function that receives a configuration dictionary and returns an instance that implements the module's Protocol.

5. **`constants.py`**: Defines module-specific constants (default values, metadata keys, etc.).

6. **`__init__.py`**: Exports the main classes, types, and functions of the module to facilitate imports.

## Configuration (`config.yaml`)

The `config.yaml` file is the central configuration file that controls which components are used and how they are configured. Each module has a corresponding section in the configuration file.

### Configuration Structure

The `config.yaml` file is organized into sections that map to each module:

```yaml
loader:
  loader_name: pymupdf          # Maps to LoaderType.PYMUPDF
  loader_config: null           # Additional loader-specific config

chunking:
  chunker_name: langchain       # Maps to ChunkerType.LANGCHAIN
  chunk_size: 1000              # Chunk size in characters
  chunk_overlap: 200            # Overlap between chunks
  method: recursive             # Chunking method

embedding:
  embed_name: huggingface       # Maps to EmbeddingModelType.HUGGINGFACE
  embed_config:                 # Additional model-specific config
    model_name: "sentence-transformers/all-MiniLM-L6-v2"

vector_store:
  store_name: chromadb          # Maps to VectorStoreType.CHROMADB
  persist_directory: ./vector_db
  collection_name: rag_collection
  store_config: null

retrieval:
  searcher_strategy: similarity # Maps to RetrieverType.SIMILARITY
  k: 5                          # Number of results to retrieve
  searcher_config: null

paths:
  pdf_path: ./data
  markdown_dir: ./data/markdown

logging:
  level: INFO
```

### How Configuration Maps to Components

The configuration values directly map to the Enum types defined in each module:

- **`loader_name`** → `LoaderType` enum (e.g., `"pymupdf"` → `LoaderType.PYMUPDF`)
- **`chunker_name`** → `ChunkerType` enum (e.g., `"langchain"` → `ChunkerType.LANGCHAIN`)
- **`embed_name`** → `EmbeddingModelType` enum (e.g., `"huggingface"` → `EmbeddingModelType.HUGGINGFACE`)
- **`store_name`** → `VectorStoreType` enum (e.g., `"chromadb"` → `VectorStoreType.CHROMADB`)
- **`searcher_strategy`** → `RetrieverType` enum (e.g., `"similarity"` → `RetrieverType.SIMILARITY`)

The system uses these values to:
1. Load the configuration from `config.yaml`
2. Map the string values to the corresponding Enum types
3. Use the Factory pattern to create instances of the selected components
4. Pass additional configuration parameters to the component constructors

### Configuration Examples

**Using HuggingFace embeddings:**
```yaml
embedding:
  embed_name: huggingface
  embed_config:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
```

**Using OpenAI embeddings:**
```yaml
embedding:
  embed_name: openai
  embed_config:
    model: "text-embedding-3-small"
    openai_api_key: "${OPENAI_API_KEY}"  # Can use environment variables
```

**Using a different chunking strategy:**
```yaml
chunking:
  chunker_name: langchain
  chunk_size: 1500
  chunk_overlap: 300
  method: character  # Options: recursive, character, token
```

**Note**: When adding a new component, the value you use in `config.yaml` must match the Enum value (the string value, not the Enum name). For example, if you add `COHERE = "cohere"` to the Enum, use `embed_name: cohere` in the config file.

## How to Add New Components

To add a new component to any module, follow these steps (we'll use the `embeddings` module as an example, but the process is identical for all modules):

### Step 1: Add the new type to the Enum

Edit `src/embeddings/types.py` and add the new type:

```python
class EmbeddingModelType(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"  # New type
```

### Step 2: Create the implementation file

Create a new file, for example `src/embeddings/cohere.py`:

```python
"""Cohere embedding model implementation."""
from __future__ import annotations

from typing import Dict, Any
from langchain_cohere import CohereEmbeddings

from .protocol import Embeddings

def create_cohere_embedding(config: Dict[str, Any]) -> Embeddings:
    """Create Cohere embedding model.
    
    Parameters
    ----------
    config
        Configuration dictionary. Common parameters include:
        - model: str (optional) - Model name
        - cohere_api_key: str (optional) - API key
        - Other parameters supported by CohereEmbeddings constructor.
    
    Returns
    -------
    Embeddings instance.
    """
    try:
        return CohereEmbeddings(**config)
    except Exception as e:
        raise ValueError(f"Failed to create Cohere embedding model: {e}") from e
```

**Important**: The function must:
- Receive a `Dict[str, Any]` as parameter
- Return an instance that implements the module's Protocol (`Embeddings` in this case)
- Handle errors appropriately

### Step 3: Register the implementation in the Factory

Edit `src/embeddings/factory.py` and add the import and registration:

```python
from .cohere import create_cohere_embedding  # Add import

# At the end of the file, register the new implementation
EmbeddingModelFactory.register(EmbeddingModelType.COHERE)(create_cohere_embedding)
```

### Step 4: Update exports (optional)

If necessary, update `src/embeddings/__init__.py` to export any constants or helpers related to the new component.

### Step 5: Configure in `config.yaml`

Add the configuration for the new component in `config.yaml`:

```yaml
embedding:
  embed_name: cohere  # Use the Enum value (must match the string value in types.py)
  embed_config:
    model: "embed-english-v3.0"
    cohere_api_key: "${COHERE_API_KEY}"  # Can use environment variables
```

### Process Summary

1. ✅ Add type to Enum in `types.py`
2. ✅ Create implementation file with `create_*` function
3. ✅ Import and register in `factory.py`
4. ✅ Configure in `config.yaml` (if applicable)

This same process applies to:
- **`chunkers/`**: Add new chunking algorithms
- **`loaders/`**: Add new loader types (DOCX, HTML, etc.)
- **`retrievers/`**: Add new retrieval strategies
- **`vector_stores/`**: Add new vector stores (Pinecone, Weaviate, etc.)

## Execution

```bash
# Build
docker-compose build

# Ingest all PDFs in data folder
docker-compose run --rm ingest python src/cli/ingest.py /app/data/

# Query
docker-compose run --rm query python src/cli/query.py "your question"

# Clean up
docker-compose down
```