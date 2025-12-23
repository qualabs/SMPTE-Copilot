# SMPTE-Copilot

An open-source AI co-pilot that ingests and indexes text, audio, and video to enable semantic, multimodal search of media archives. The prototype provides modular ingestion, a chat-based retrieval pipeline, transparent citations, and tiered access for public users, members, and staff.

## Execution

### CLI Usage

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

### API Server (OpenAI-Compatible)

The project includes an OpenAI-compatible REST API server that can be integrated with tools like OpenWebUI, or any OpenAI-compatible client.

```bash
# Start the API server
docker-compose up api

# API will be available at http://localhost:8000
# OpenAI-compatible endpoint: http://localhost:8000/v1/chat/completions
```

**Test the API:**

```bash
# Using curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smpte-copilot",
    "messages": [{"role": "user", "content": "What is SMPTE ST 2110?"}]
  }'
```

## Project Structure

The project is organized into modular components that follow a consistent pattern. Each module implements the Factory pattern to enable easy extension and addition of new components.

```
SMPTE-Copilot/
├── src/
│   ├── api/               # REST API server
│   ├── chunkers/          # Module for splitting documents into chunks
│   ├── embeddings/        # Module for embedding models
│   ├── llms/              # Module for LLM models
│   ├── loaders/           # Module for loading documents from various sources
│   ├── retrievers/        # Module for document retrieval
│   ├── vector_stores/     # Module for vector storage
│   ├── pipeline/          # Pipeline orchestration and execution
│   ├── config/            # Project configuration
│   └── cli/               # Command-line interfaces
├── data/                  # Data and documents to process
├── scripts/               # Utility scripts (test_api.py, query_api.py)
├── config.yaml           # Main configuration file
├── docker-compose.yml    # Docker configuration
├── Dockerfile            # Dockerfile for CLI tools
└── Dockerfile.api        # Dockerfile for API server
```

## Architecture Patterns

The project uses two main architectural patterns that enable modularity and extensibility:

1. **Factory Pattern with Dynamic Registry**: For component creation and registration
2. **Pipeline Pattern**: For orchestrating sequential processing steps

### Module Architecture

All main modules (`chunkers`, `embeddings`, `llms`, `loaders`, `retrievers`, `vector_stores`) follow the same architectural structure based on the Factory pattern. This consistency facilitates code understanding and the incorporation of new components.

#### Module Structure (Example: `embeddings/`)

```
embeddings/
├── __init__.py           # Exports main classes and types
├── protocol.py           # Defines the Protocol interface that all components must implement
├── types.py              # Defines the Enum with available types
├── factory.py            # Implements the Factory pattern with dynamic registry
├── constants.py          # Module-specific constants (optional)
├── helpers.py            # Helper functions (optional)
├── huggingface.py        # Specific implementation: HuggingFace embeddings
└── openai.py             # Specific implementation: OpenAI embeddings
```

**Main components:**

1. **`protocol.py`**: Defines a Protocol (interface) that specifies the methods all implementations must provide. This ensures compatibility and allows swapping implementations without changing the rest of the code.

2. **`types.py`**: Contains an Enum that lists all available component types in the module (e.g., `EmbeddingModelType.HUGGINGFACE`, `EmbeddingModelType.OPENAI`).

3. **`factory.py`**: Implements the Factory pattern with a dynamic registry. Allows registering new implementations and creating them by type. The factory maintains a dictionary that maps types to creation functions.

4. **Implementation files** (e.g., `huggingface.py`, `openai.py`): Each file contains a `create_*` function that receives a configuration dictionary and returns an instance that implements the module's Protocol.

5. **`constants.py`**: Defines module-specific constants (default values, metadata keys, etc.).

6. **`__init__.py`**: Exports the main classes, types, and functions of the module to facilitate imports.

### Dynamic Factory Pattern with Registry

The project uses a **dynamic Factory pattern with an internal registry** to enable runtime registration of component implementations. This pattern provides maximum flexibility and extensibility without requiring modifications to the factory class when adding new implementations.

#### How It Works

Each Factory class maintains an internal `_registry` dictionary that maps component types (Enum values) to factory functions:

```python
class EmbeddingModelFactory:
    """Factory for creating embedding models. Easily extensible."""

    # Class variable: shared registry across all instances
    _registry: ClassVar[dict[EmbeddingModelType, Callable[[dict[str, Any]], Embeddings]]] = {}
```

#### Registration Mechanism

The Factory provides a `register` method that acts as a decorator, allowing implementations to be registered dynamically:

```python
@classmethod
def register(cls, model_type: EmbeddingModelType):
    """Register a new embedding model factory.

    Parameters
    ----------
    model_type
        Type to register the model under.
    """
    def decorator(factory_func: Callable[[dict[str, Any]], Embeddings]):
        cls._registry[model_type] = factory_func
        return factory_func
    return decorator
```

#### Registration Process

Implementations are registered at module load time (when the factory module is imported):

```python
# At the end of factory.py
EmbeddingModelFactory.register(EmbeddingModelType.HUGGINGFACE)(create_huggingface_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.OPENAI)(create_openai_embedding)
```

This registration happens automatically when the module is imported, populating the registry before any `create()` calls are made.

#### Creation Process

When `create()` is called, the factory looks up the type in the registry and calls the corresponding factory function:

```python
@classmethod
def create(cls, model_type: EmbeddingModelType, **kwargs) -> Embeddings:
    """Create an embedding model by type."""
    if model_type not in cls._registry:
        available = ", ".join(t.value for t in cls._registry)
        raise ValueError(
            f"Unknown model: {model_type}. "
            f"Available models: {available}"
        )
    return cls._registry[model_type](kwargs)
```

#### Complete Example: EmbeddingModelFactory

```python
class EmbeddingModelFactory:
    """Factory for creating embedding models. Easily extensible."""

    # Internal registry: maps EmbeddingModelType -> factory function
    _registry: ClassVar[dict[EmbeddingModelType, Callable[[dict[str, Any]], Embeddings]]] = {}

    @classmethod
    def register(cls, model_type: EmbeddingModelType):
        """Register a new embedding model factory."""
        def decorator(factory_func: Callable[[dict[str, Any]], Embeddings]):
            cls._registry[model_type] = factory_func
            return factory_func
        return decorator

    @classmethod
    def create(cls, model_type: EmbeddingModelType, **kwargs) -> Embeddings:
        """Create an embedding model by type."""
        if model_type not in cls._registry:
            available = ", ".join(t.value for t in cls._registry)
            raise ValueError(
                f"Unknown model: {model_type}. "
                f"Available models: {available}"
            )
        return cls._registry[model_type](kwargs)

# Register implementations at module load time
EmbeddingModelFactory.register(EmbeddingModelType.HUGGINGFACE)(create_huggingface_embedding)
EmbeddingModelFactory.register(EmbeddingModelType.OPENAI)(create_openai_embedding)
```

#### Benefits of the Registry Pattern

1. **Zero Factory Modification**: Adding a new implementation doesn't require modifying the Factory class
2. **Runtime Flexibility**: Registry is populated at import time, allowing dynamic discovery
3. **Type Safety**: Registry is strongly typed with `ClassVar` and type hints
4. **Error Messages**: Clear error messages listing available types when an unknown type is requested
5. **Testability**: Easy to mock or replace implementations in tests by manipulating the registry
6. **Extensibility**: Third-party code can register new implementations without modifying core code

#### Registry Flow Diagram

```
Module Import
    ↓
Factory class definition loaded
    ↓
Registry dictionary initialized (empty)
    ↓
Registration statements executed
    ↓
Registry populated: {Type1: func1, Type2: func2, ...}
    ↓
Factory.create(Type1, **config) called
    ↓
Lookup Type1 in registry
    ↓
Call registered function: func1(config)
    ↓
Return instance
```

## Pipeline Pattern Architecture

The project uses a **Pipeline Pattern** to orchestrate sequential processing steps. This pattern provides a clean separation of concerns, makes the codebase highly extensible, and allows easy addition of new processing steps without modifying existing code.

### Overview

The pipeline pattern consists of three main components:

1. **Context**: A data structure that holds the state as it flows through the pipeline
2. **Steps**: Individual processing units that transform the context
3. **Executor**: Orchestrates the execution of steps sequentially

### Ingestion Pipeline

The ingestion pipeline (`ingest.py`) processes documents through four sequential steps:

```
Load → Chunk → Embed → Save
```

**Pipeline Flow:**

1. **LoadStep**: Converts media files (PDF, images, videos, audio) to Markdown format

   - Input: `file_path` in `IngestionContext`
   - Output: Sets `markdown_path` and `raw_text` in context

2. **ChunkStep**: Splits the Markdown text into smaller chunks

   - Input: `markdown_path` from LoadStep
   - Output: Sets `chunks` (list of Document objects) in context

3. **EmbeddingGenerationStep**: Generates embeddings for each chunk

   - Input: `chunks` from ChunkStep
   - Output: Updates `chunks` with embeddings in metadata and sets `vectors`

4. **SaveStep**: Stores chunks with embeddings in the vector database
   - Input: `chunks` with embeddings from EmbeddingGenerationStep
   - Output: Persists data to vector store

**Implementation Example:**

```python
from src.pipeline import IngestionContext, PipelineExecutor
from src.pipeline.steps import LoadStep, ChunkStep, EmbeddingGenerationStep, SaveStep

context = IngestionContext(file_path=file_path)

steps = [
    LoadStep(loader),
    ChunkStep(chunker),
    EmbeddingGenerationStep(embedding_model, model_name),
    SaveStep(vector_store),
]

executor = PipelineExecutor(steps)
context = executor.execute(context)
```

### Query Pipeline

The query pipeline (`query.py`) processes user queries through three sequential steps:

```
QueryEmbedding → Retrieve → Generate
```

**Pipeline Flow:**

1. **QueryEmbeddingStep**: Generates an embedding vector for the user query

   - Input: `user_query` in `QueryContext`
   - Output: Sets `query_vector` in context

2. **RetrieveStep**: Retrieves relevant documents from the vector store

   - Input: `user_query` (uses query directly, not the vector)
   - Output: Sets `retrieved_docs` (list of tuples with Document and score) in context

3. **GenerateStep**: Generates a response using an LLM based on the retrieved documents
   - Input: `retrieved_docs` from RetrieveStep
   - Output: Sets `response` and `citations` in context

**Implementation Example:**

```python
from src.pipeline import QueryContext, PipelineExecutor
from src.pipeline.steps import QueryEmbeddingStep, RetrieveStep

context = QueryContext(user_query=query)

steps = [
    QueryEmbeddingStep(embedding_model),
    RetrieveStep(retriever),
    GenerateStep(llm),
]

executor = PipelineExecutor(steps)
context = executor.execute(context)
```

### Pipeline Context

Each pipeline uses a context object that extends `PipelineContext`:

- **`IngestionContext`**: Tracks document state through ingestion

  - `file_path`: Path to the source file
  - `markdown_path`: Path to generated Markdown file
  - `chunks`: List of document chunks
  - `vectors`: List of embedding vectors
  - `status`: Pipeline execution status (PENDING, RUNNING, COMPLETED, FAILED)
  - `error`: Error message if pipeline failed

- **`QueryContext`**: Tracks query state through retrieval
  - `user_query`: Original user query string
  - `query_vector`: Embedding vector for the query
  - `retrieved_docs`: Retrieved documents with similarity scores
  - `response`: Generated response from LLM
  - `citations`: List of citations for the response
  - `status`: Pipeline execution status
  - `error`: Error message if pipeline failed

### Extensibility: Adding New Steps

The pipeline pattern makes it extremely easy to add new processing steps. For example, to add a **re-ranking step** to the query pipeline:

#### Step 1: Create the Re-ranking Step

Create `src/pipeline/steps/rerank_step.py`:

```python
class RerankStep:
    """Step that re-ranks retrieved documents using a re-ranker model."""

    def __init__(self, reranker):
        """Initialize the re-rank step.

        Parameters
        ----------
        reranker
            Re-ranker model instance.
        """
        self.reranker = reranker

    def run(self, context: QueryContext) -> None:
        """Re-rank retrieved documents.

        Parameters
        ----------
        context
            Query context with retrieved_docs set.
        """
        ...
```

#### Step 2: Export the Step

Add to `src/pipeline/steps/__init__.py`:

```python
from .rerank_step import RerankStep

__all__ = [
    # ... existing steps
    "RerankStep",
]
```

#### Step 3: Use in Pipeline

Update `src/cli/query.py`:

```python
from src.pipeline.steps import QueryEmbeddingStep, RetrieveStep, RerankStep

steps = [
    QueryEmbeddingStep(embedding_model),
    RetrieveStep(retriever),
    RerankStep(reranker),  # New step added here
]

executor = PipelineExecutor(steps)
context = executor.execute(context)
```

That's it! The new step is seamlessly integrated into the pipeline. The executor will:

1. Execute steps in order
2. Stop if any step marks the context as failed
3. Handle errors appropriately

### Benefits of the Pipeline Pattern

1. **Modularity**: Each step is independent and can be tested in isolation
2. **Extensibility**: Add new steps without modifying existing code
3. **Flexibility**: Reorder steps or create different pipeline configurations
4. **Error Handling**: Centralized error handling through the executor
5. **State Management**: Context object provides clear state tracking
6. **Composability**: Mix and match steps to create different pipelines

### Pipeline Execution Flow

```
1. Create context with initial data
2. Create list of steps
3. Create PipelineExecutor with steps
4. Execute pipeline:
   - Mark context as RUNNING
   - For each step:
     - Check if context is FAILED (stop if so)
     - Execute step.run(context)
     - Step modifies context
   - If still RUNNING, mark as COMPLETED
5. Return context with final state
```

## Configuration (`config.yaml`)

The `config.yaml` file is the central configuration file that controls which components are used and how they are configured. Each module has a corresponding section in the configuration file.

### Configuration Structure

The `config.yaml` file is organized into sections that map to each module:

```yaml
loader:
  file_type_mapping: # Map file extensions to loader types
    .pdf:
      loader_name: pymupdf # PDF files use pymupdf loader
      loader_config: null # Optional loader-specific configuration

chunking:
  chunker_name: langchain # Maps to ChunkerType.LANGCHAIN
  chunk_size: 1000 # Chunk size in characters
  chunk_overlap: 200 # Overlap between chunks
  method: recursive # Chunking method

embedding:
  embed_name: huggingface # Maps to EmbeddingModelType.HUGGINGFACE
  embed_config: # Additional model-specific config
    model_name: "sentence-transformers/all-MiniLM-L6-v2"

llm:
  llm_name: gemini
  llm_config:
    model: gemini-2.5-flash
    # api_key: "${GEMINI_API_KEY}"

vector_store:
  store_name: chromadb # Maps to VectorStoreType.CHROMADB
  persist_directory: ./vector_db
  collection_name: rag_collection
  store_config: null

retrieval:
  searcher_strategy: similarity # Maps to RetrieverType.SIMILARITY
  k: 5 # Number of results to retrieve
  searcher_config: null

paths:
  input_path: ./data # Default path for input media files
  markdown_dir: ./data/markdown # Directory for markdown output

logging:
  level: INFO
```

### How Configuration Maps to Components

The configuration values directly map to the Enum types defined in each module:

- **`loader.file_type_mapping`** → Maps file extensions (e.g., `.pdf`) to loader configurations. Each entry contains `loader_name` (e.g., `"pymupdf"` → `LoaderType.PYMUPDF`) and optional `loader_config`
- **`chunker_name`** → `ChunkerType` enum (e.g., `"langchain"` → `ChunkerType.LANGCHAIN`)
- **`embed_name`** → `EmbeddingModelType` enum (e.g., `"huggingface"` → `EmbeddingModelType.HUGGINGFACE`)
- **`store_name`** → `VectorStoreType` enum (e.g., `"chromadb"` → `VectorStoreType.CHROMADB`)
- **`searcher_strategy`** → `RetrieverType` enum (e.g., `"similarity"` → `RetrieverType.SIMILARITY`)

The system uses these values to:

1. Load the configuration from `config.yaml`
2. Map the string values to the corresponding Enum types
3. Use the Factory pattern to create instances of the selected components
4. Pass additional configuration parameters to the component constructors

**Note on Loader Configuration**: The `loader.file_type_mapping` allows you to configure different loaders for different file types. This enables the system to support multiple file formats (PDF, images, videos, audio) with appropriate loaders for each type. When adding support for a new file type, add an entry to `file_type_mapping` with the file extension as the key.

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
    openai_api_key: "${OPENAI_API_KEY}" # Can use environment variables
```

**Using a different chunking strategy:**

```yaml
chunking:
  chunker_name: langchain
  chunk_size: 1500
  chunk_overlap: 300
  method: character # Options: recursive, character, token
```

**Configuring loaders for different file types:**

```yaml
loader:
  file_type_mapping:
    .pdf:
      loader_name: pymupdf
      loader_config: null
    # When other loaders are added, you can configure them like:
    # .mp4:
    #   loader_name: video_loader
    #   loader_config:
    #     extract_audio: true
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

# At the end of the file, register the new implementation in the registry
EmbeddingModelFactory.register(EmbeddingModelType.COHERE)(create_cohere_embedding)
```

The registration happens automatically at **module import time** (see [Dynamic Factory Pattern with Registry](#dynamic-factory-pattern-with-registry) for details on how the registry works).

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

### Process Summary

1. Add type to Enum in `types.py`
2. Create implementation file with `create_*` function
3. Import and register in `factory.py` (registry populated automatically)
4. Configure in `config.yaml` (if applicable)

This same process applies to:

- **`chunkers/`**: Add new chunking algorithms
- **`loaders/`**: Add new loader types (DOCX, HTML, etc.)
- **`retrievers/`**: Add new retrieval strategies
- **`vector_stores/`**: Add new vector stores (Pinecone, Weaviate, etc.)

## Command-Line Interface (CLI)

The project provides two main CLI commands for ingesting documents and querying the vector database.

### `ingest.py` - Document Ingestion

The `ingest.py` command processes media files and adds them to the vector database using the [ingestion pipeline](#ingestion-pipeline). It is designed to support multiple file types including PDFs, images, videos, and audio files (currently supports PDFs, with multimodal support planned).

**Usage:**

```bash
# Ingest a single file (currently supports PDF)
python src/cli/ingest.py /path/to/document.pdf

# Ingest all supported files in a directory
python src/cli/ingest.py /path/to/directory/

# Using default path from config.yaml (uses paths.input_path)
python src/cli/ingest.py
```

**What it does:**

- Accepts media files or directories containing supported file types
- Currently supports PDF files; future versions will support images, videos, and audio
- Automatically selects the appropriate loader based on file extension using `loader.file_type_mapping` in `config.yaml`
- Uses components configured in `config.yaml` (loader, chunker, embedding model, vector store)
- Executes the ingestion pipeline: Load → Chunk → Embed → Save (see [Ingestion Pipeline](#ingestion-pipeline) for details)
- Saves processed content (e.g., Markdown files) to the configured output directory (`paths.markdown_dir`)
- Provides detailed logging of each step in the pipeline

**Output:**

- Processed content files saved to the configured output directory
- Vector database populated with embedded document chunks
- Summary of processed files, chunk counts, and database location

### `query.py` - Document Querying

The `query.py` command searches the vector database for documents similar to a given query using the [query pipeline](#query-pipeline).

**Usage:**

```bash
# Query with a question
python src/cli/query.py "What is the main topic of the document?"

# Query with multiple words (all arguments are combined)
python src/cli/query.py your question here
```

**What it does:**

1. Validates that the vector database exists (must run `ingest.py` first)
2. Creates the embedding model, vector store, and retriever using `config.yaml` settings
3. Executes the query pipeline: QueryEmbedding → Retrieve (see [Query Pipeline](#query-pipeline) for details)
4. Displays results with similarity scores and document content

**Output:**

- List of relevant documents ranked by similarity score
- Each result includes:
  - Similarity score (higher = more similar)
  - Document content (chunk text)
  - Metadata (source file, page numbers, etc.)

**Important**: The embedding model used for querying must match the one used during ingestion to ensure accurate similarity search.
