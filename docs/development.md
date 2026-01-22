# Development Guide

This guide explains how to extend SMPTE-Copilot by adding new components to any module.

## Table of Contents

- [How to Add New Components](#how-to-add-new-components)
- [Process Summary](#process-summary)
- [Module-Specific Notes](#module-specific-notes)

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

# At the end of the file, register the new implementation in the registry
EmbeddingModelFactory.register(EmbeddingModelType.COHERE)(create_cohere_embedding)
```

The registration happens automatically at **module import time** (see [Dynamic Factory Pattern with Registry](architecture.md#dynamic-factory-pattern-with-registry) for details on how the registry works).

### Step 4: Update exports (optional)

If necessary, update `src/embeddings/__init__.py` to export any constants or helpers related to the new component.

### Step 5: Configure in `config.yaml`

Add the configuration for the new component in `config.yaml`:

```yaml
embedding:
  embed_name: cohere # Use the Enum value (must match the string value in types.py)
  embed_config:
    model: "embed-english-v3.0"
    cohere_api_key: "${COHERE_API_KEY}" # Can use environment variables
```

## Process Summary

1. Add type to Enum in `types.py`
2. Create implementation file with `create_*` function
3. Import and register in `factory.py` (registry populated automatically)
4. Configure in `config.yaml` (if applicable)

## Module-Specific Notes

This same process applies to:
- **`chunkers/`**: Add new chunking algorithms
- **`loaders/`**: Add new loader types (DOCX, HTML, etc.)
- **`retrievers/`**: Add new retrieval strategies
- **`vector_stores/`**: Add new vector stores (Pinecone, Weaviate, etc.)

For more information about the architecture patterns used, see the [Architecture](architecture.md) documentation. To understand how to configure new components, see the [Configuration](configuration.md) guide.
