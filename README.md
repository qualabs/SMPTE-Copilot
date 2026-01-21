# SMPTE-Copilot

An open-source AI co-pilot that ingests and indexes text, audio, and video to enable semantic, multimodal search of media archives. The prototype provides modular ingestion, a chat-based retrieval pipeline, transparent citations, and tiered access for public users, members, and staff.

## Execution

```bash
# Build
docker compose build

# Start Qdrant vector database
docker compose up qdrant

# Ingest all PDFs in data folder
docker compose run --rm ingest python src/cli/ingest.py /app/data/

# Query
docker compose run --rm query python src/cli/query.py "your question"

# Clean up
docker compose down
```

### API Server (OpenAI-Compatible)

The project includes an OpenAI-compatible REST API server that can be integrated with tools like OpenWebUI, or any OpenAI-compatible client.

```bash
# Start the API server
docker compose up api

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
### OpenWebUI Integration

SMPTE-Copilot can be used with **OpenWebUI** as a chat interface via its OpenAI-compatible API. Once the `api` and `openwebui` services are running, access the UI at **http://localhost:3000**. The backend is automatically configured through `OPENAI_API_BASE_URL` to use the local RAG API (`/v1/chat/completions`). 

To start OpenWebUI with SMPTE-Copilot:

```bash
docker compose up openwebui
```

#### Enabling Clickable Citations

By default, the integration uses the standard OpenAI-compatible endpoint. To enable **clickable citations** that show the source chunks when clicked, you need to install the SMPTE Copilot Pipe in OpenWebUI:

1. **Access Admin Settings**: Log into OpenWebUI as an admin and go to **Admin Panel** → **Settings** → **Functions**

2. **Create New Function**: Click the **+** button to create a new function

3. **Copy Pipe Code**: Copy the entire contents of [`src/openwebui/smpte_pipe.py`](src/openwebui/smpte_pipe.py) and paste it into the function editor

4. **Save and Enable**: Save the function and ensure it's enabled

5. **Configure Valves** (if needed): Click the gear icon on the function to configure:
   - `SMPTE_API_BASE_URL`: The backend URL (default: `http://api:8000`)
   - `REQUEST_TIMEOUT`: Request timeout in seconds (default: `120`)

6. **Use the Pipe**: In your chat, select the model **"SMPTE Copilot RAG"** instead of the standard "smpte-copilot" model

When using the Pipe, citations like `[1]`, `[2]`, etc. in the response will be clickable, showing a popup with the source document name, page number, and the actual chunk content that was retrieved.

**Note**: The Pipe uses the `/v1/rag/query` endpoint which returns both the response and citation metadata, while the standard OpenAI-compatible endpoint (`/v1/chat/completions`) only returns the response text.

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
│   ├── config/            # Project configuration
│   └── cli/               # Command-line interfaces
├── data/                  # Data and documents to process
├── config.yaml           # Main configuration file
└── docker-compose.yml    # Docker configuration
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

The ingestion pipeline (`ingest.py`) processes documents through sequential steps. Each step can be enabled or disabled via configuration (see [Configurable Pipelines](#configurable-pipelines)):

```
Load → Preprocess → Chunk → Embed → Save
```

**Pipeline Flow:**

1. **LoadStep**: Converts media files (PDF, images, videos, audio) to Markdown format
   - Input: `file_path` in `IngestionContext`
   - Output: Sets `markdown_path` and `raw_text` in context

2. **PreprocessStep**: Removes repeated headers, footers, and page numbers
   - Input: `raw_text` from LoadStep
   - Output: Updates `raw_text` and `markdown_path` with cleaned content
   - Configurable via `preprocessing` section in `config.yaml`

3. **ChunkStep**: Splits the Markdown text into smaller chunks
   - Input: `markdown_path` from LoadStep/PreprocessStep
   - Output: Sets `chunks` (list of Document objects) in context

4. **EmbeddingGenerationStep**: Generates embeddings for each chunk
   - Input: `chunks` from ChunkStep
   - Output: Updates `chunks` with embeddings in metadata and sets `vectors`

5. **SaveStep**: Stores chunks with embeddings in the vector database
   - Input: `chunks` with embeddings from EmbeddingGenerationStep
   - Output: Persists data to vector store

**Implementation Example:**

```python
from src.pipeline import IngestionContext, PipelineExecutor
from src.pipeline.steps import (
    LoadStep,
    PreprocessStep,
    ChunkStep,
    EmbeddingGenerationStep,
    SaveStep,
)
context = IngestionContext(file_path=file_path)

# Steps are built dynamically based on pipeline configuration
# Only enabled steps are included
steps = []
if config.pipeline.ingestion.load_enabled:
    steps.append(LoadStep(loader))
if config.pipeline.ingestion.preprocess_enabled:
    steps.append(PreprocessStep(preprocessor))
if config.pipeline.ingestion.chunk_enabled:
    steps.append(ChunkStep(chunker))
if config.pipeline.ingestion.save_enabled:
    # Save step includes both embedding generation and database storage
    steps.append(EmbeddingGenerationStep(embedding_model, model_name))
    steps.append(SaveStep(vector_store))

executor = PipelineExecutor(steps)
context = executor.execute(context)
```

### Query Pipeline

The query pipeline (`query.py`) processes user queries through sequential steps. Each step can be enabled or disabled via configuration (see [Configurable Pipelines](#configurable-pipelines)):

```
QueryEmbedding → Retrieve → [Rerank] → [AccessControl] → Generate
```

**Pipeline Flow:**

1. **QueryEmbeddingStep**: Generates an embedding vector for the user query
   - Input: `user_query` in `QueryContext`
   - Output: Sets `query_vector` in context

2. **RetrieveStep**: Retrieves relevant documents from the vector store
   - Input: `user_query` (uses query directly, not the vector)
   - Output: Sets `retrieved_docs` (list of tuples with Document and score) in context
   - When `notify_on_denied_access: false`: Applies database-level access control filtering
   - When `notify_on_denied_access: true`: Retrieves all matching documents without filtering

3. **RerankStep** (optional): Reranks retrieved documents using a cross-encoder
   - Only active when `rerank_enabled: true`
   - Input: `retrieved_docs` from RetrieveStep
   - Output: Updates `retrieved_docs` with reordered documents based on cross-encoder relevance scores
   - Improves precision by using more sophisticated relevance assessment

4. **AccessControlStep** (optional): Separates accessible and restricted documents
   - Only active when `notify_on_denied_access: true`
   - Input: `retrieved_docs` from RetrieveStep/RerankStep
   - Output: Updates `retrieved_docs` with only accessible documents, sets `restricted_docs` and `has_restricted_content`

5. **GenerateStep**: Generates a response using an LLM based on the retrieved documents
   - Input: `retrieved_docs` from previous steps
   - Output: Sets `llm_response` and `citations` in context
   - When `has_restricted_content: true`: Appends access denial notification to response

**Implementation Example:**

```python
from src.pipeline import QueryContext, PipelineExecutor
from src.pipeline.steps import (
    QueryEmbeddingStep,
    RetrieveStep,
    RerankStep,
    AccessControlStep,
    GenerationStep,
)

context = QueryContext(user_query=query)

# Steps are built dynamically based on pipeline configuration
# Only enabled steps are included
steps = []
if config.pipeline.query.retrieve_enabled:
    steps.append(QueryEmbeddingStep(embedding_model))
    steps.append(RetrieveStep(retriever))
if config.pipeline.query.rerank_enabled:
    steps.append(RerankStep(reranker))
if config.access_control.notify_on_denied_access:
    steps.append(AccessControlStep())
if config.pipeline.query.generation_enabled:
    steps.append(GenerationStep(llm))

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
  - `llm_response`: Generated response from LLM
  - `citations`: List of citations for the response
  - `user_role`: User's role for access control
  - `role_mapping`: Role-to-tags mapping dictionary
  - `restricted_docs`: List of restricted document metadata (when `notify_on_denied_access: true`)
  - `has_restricted_content`: Flag indicating if restricted content was found
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
from .access_control_step import AccessControlStep

__all__ = [
    # ... existing steps
    "RerankStep",
    "AccessControlStep",
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

The `config.yaml` file is organized into sections that map to each module. See [`config-example.yaml`](./config-example.yaml) for a complete example with all available options and detailed comments.

**Main configuration sections:**
- `input_source`: Input source type (local or S3) and connection settings
- `loader`: File type to loader mapping for document processing
- `preprocessing`: Text preprocessing options (e.g., duplicate removal)
- `chunking`: Document chunking strategy and parameters
- `embedding`: Embedding model selection and configuration
- `llm`: LLM model for answer generation
- `vector_store`: Vector database selection and connection settings
- `retrieval`: Retrieval strategy and parameters
- `reranking`: Cross-encoder reranking configuration
- `paths`: Input and output directory paths
- `logging`: Log level configuration
- `access_control`: Role-based access control settings
- `pipeline`: Enable/disable individual pipeline steps

### How Configuration Maps to Components

The configuration values directly map to the Enum types defined in each module:

- **`source_type`** → `InputSourceType` enum (e.g., `"S3"` -> `InputSourceType.S3`)
- **`loader.file_type_mapping`** → List of loader configurations. Each entry contains `extensions` (list of file extensions like `[.pdf, .docx]`), `loader_name` (e.g., `"pymupdf"` → `LoaderType.PYMUPDF`), and optional `loader_config`. Multiple extensions can share the same loader configuration to avoid repetition.
- **`preprocessing_name`** → `PreprocessorType` enum (e.g., `"rapidfuzz"` → `PreprocessorType.RAPIDFUZZ`)
- **`chunker_name`** → `ChunkerType` enum (e.g., `"langchain"` → `ChunkerType.LANGCHAIN`)
- **`embed_name`** → `EmbeddingModelType` enum (e.g., `"huggingface"` → `EmbeddingModelType.HUGGINGFACE`)
- **`store_name`** → `VectorStoreType` enum (e.g., `"chromadb"` → `VectorStoreType.CHROMADB`, `"qdrant"` → `VectorStoreType.QDRANT`)
- **`searcher_strategy`** → `RetrieverType` enum (e.g., `"similarity"` → `RetrieverType.SIMILARITY`)
- **`reranker_name`** → `RerankerType` enum (e.g., `"gemini"` → `RerankerType.GEMINI`)
- **`access_control`** → Access control configuration (see [Access Control System](#access-control-system) for details)
  - `notify_on_denied_access`: When `true`, enables notification mode that informs users about restricted documents

The system uses these values to:
1. Load the configuration from `config.yaml`
2. Map the string values to the corresponding Enum types
3. Use the Factory pattern to create instances of the selected components
4. Pass additional configuration parameters to the component constructors

**Note on Loader Configuration**: The `loader.file_type_mapping` allows you to configure different loaders for different file types. This enables the system to support multiple file formats (PDF, images, videos, audio) with appropriate loaders for each type. The format uses a list where each entry has an `extensions` list, allowing multiple extensions to share the same loader configuration. When adding support for a new file type, add the extension to an existing entry's `extensions` list (if it uses the same loader) or create a new entry.

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
  chunker_config:
    chunk_size: 1500
    chunk_overlap: 300
    method: character # Options: recursive, character, token
```

**Using hybrid chunking (semantic + token-based):**

```yaml
chunking:
  chunker_name: hybrid
  chunker_config:
    max_tokens: 2000  # Maximum tokens per chunk (default: 2000)
    merge_peers: false # Whether to merge peer chunks (default: false)
```

**Configuring loaders for different file types:**

```yaml
loader:
  file_type_mapping:
    - extensions: [.pdf]  # Single extension entry
      loader_name: pymupdf
      loader_config: null
    - extensions: [.docx]  # Multiple extensions can share the same loader config
      loader_name: docling
      loader_config: 
        llm_api_key: # LLM key for used for image description
        llm_endpoint: https://generativelanguage.googleapis.com/v1beta/openai/chat/completions # LLM endpoint for image description
        llm_model: gemini-2.5-flash # LLM Model for image description
        image_description_prompt: "Analyze the image exhaustively. Do not summarize; extract details." # Prompt used to tailor the LLM image description on documents
    # When other loaders are added, you can configure them like:
    # - extensions: [.mp4, .avi, .mov]  # Multiple video formats can share the same loader
    #   loader_name: whisper
    #   loader_config:
    #     model_name: base
    #     device: cpu
```

**Note**: When adding a new component, the value you use in `config.yaml` must match the Enum value (the string value, not the Enum name). For example, if you add `COHERE = "cohere"` to the Enum, use `embed_name: cohere` in the config file.

## Access Control System

SMPTE-Copilot includes a tag-based access control system that allows you to control which documents users can access based on their roles. This system works by:

1. **Tagging documents during ingestion** with access tags (e.g., `["Public"]`, `["Finance", "Public"]`)
2. **Mapping user roles to authorized tags** via a role mapping file
3. **Filtering query results** to only return documents the user is authorized to access

### How It Works

The access control system uses a **role-to-tags mapping** approach:

- **Documents** are tagged with `access_tags` during ingestion (e.g., `["Finance", "Public"]`)
- **Users** are assigned roles (e.g., `"Finance_Manager"`, `"Public"`)
- **Roles** are mapped to authorized tags via `access_mapping.json` (e.g., `"Finance_Manager"` → `["Finance", "Public"]`)
- **Queries** automatically filter results to only include documents where at least one of the document's `access_tags` matches one of the user's authorized tags

### Access Mapping File

The `access_mapping.json` file contains two mappings:
- **folders**: Maps folder names to access tags (used during ingestion)
- **roles**: Maps user roles to authorized tags (used during queries)

Users can access documents that have at least one tag matching their authorized tags.

**Example `access_mapping.json`:**

```json
{
  "folders": {
    "Finance": ["Finance", "Public"],
    "HR": ["HR", "Public"],
    "Admin": ["Finance", "HR", "Public", "Admin"],
    "Protected": ["Protected"]
  },
  "roles": {
    "Public": ["Public"],
    "Finance_Manager": ["Finance", "Public"],
    "HR_Manager": ["HR", "Public"],
    "Admin": ["Finance", "HR", "Public", "Admin"],
    "Protected": ["Protected"]
  }
}
```

**How it works:**
- A user with role `"Finance_Manager"` can access documents tagged with `"Finance"` OR `"Public"`
- A user with role `"Public"` can only access documents tagged with `"Public"`
- A user with role `"Admin"` can access documents with any of: `"Finance"`, `"HR"`, `"Public"`, or `"Admin"`

### User Role Resolver

The **User Role Resolver** determines which role a user has based on their identity (email address). This is essential for integrating with authentication systems like OpenWebUI with Google OAuth.

#### User Mapping File

The `user_mapping.json` file maps user email addresses to their roles:

```json
{
  "users": {
    "admin@company.com": "Admin",
    "finance.manager@company.com": "Finance_Manager",
    "hr.manager@company.com": "HR_Manager"
  },
  "default_role": "Public"
}
```

- **users**: Maps email addresses to role names (must match roles defined in `access_mapping.json`)
- **default_role**: Role assigned to users not found in the mapping

#### Configuration

Configure the user resolver in `config.yaml`:

```yaml
user_resolver:
  resolver_name: json            # User role resolver type (currently only 'json' supported)
  resolver_config:
    mapping_file: "./user_mapping.json"  # Path to the user-to-role mapping file
```

#### How It Works with OpenWebUI

When using the API with OpenWebUI (with `ENABLE_FORWARD_USER_INFO_HEADERS=true`):

1. User authenticates via Google OAuth in OpenWebUI
2. OpenWebUI forwards the user's email in the `X-OpenWebUI-User-Email` header
3. The API server uses the user resolver to look up the email in `user_mapping.json`
4. If found, the mapped role is used; otherwise, `default_role` is applied
5. The role is then expanded to authorized tags via `access_mapping.json`

**Important:** The user mapping is loaded once at server startup. If you modify `user_mapping.json`, you must restart the API server for changes to take effect:

```bash
docker compose restart api
```

### Access Denial Notification Mode

The system supports two modes for handling restricted documents during queries:

#### Silent Mode (Default)

When `notify_on_denied_access: false` (default):
- Uses efficient database-level filtering (Qdrant/ChromaDB)
- Restricted documents are never retrieved from the database
- Users only see documents they have permission to access
- Best performance, recommended for production

#### Notification Mode

When `notify_on_denied_access: true`:
- Retrieves all matching documents regardless of access permissions
- Separates documents into accessible and restricted categories
- Appends a notification to the response listing restricted documents the user cannot access
- Users are informed about additional relevant content that requires higher permissions

**Example notification appended to response:**
```
---
**Note:** 2 additional document(s) matched your query but you lack permission to access them:
- confidential-report.pdf (requires: Finance)
- internal-memo.pdf (requires: HR, Admin)
```

**Configuration:**
```yaml
access_control:
  default_user_role: "Public"
  access_mapping_file: "./access_mapping.json"
  notify_on_denied_access: true  # Enable notification mode
```

### Usage Examples

#### During Ingestion

Documents receive `default_access_tags` from the configuration only if they don't already have `access_tags` in their metadata. This allows loaders or other sources to provide document-specific tags that take precedence over the default configuration.

**Current behavior:**
- If a document's chunks already have `access_tags` in their metadata, those tags are preserved
- If a document's chunks don't have `access_tags`, they receive the tags specified in `default_access_tags`
- Tags are stored in the `access_tags` metadata field of each chunk

#### During Querying

When querying, the system automatically:
1. **Resolves the user's role** using the user resolver (from `user_mapping.json` based on email) or falls back to `default_user_role`
2. Loads the role mapping from `access_mapping_file`
3. Expands the user's role to authorized tags
4. Filters query results based on the `notify_on_denied_access` setting:
   - **Silent mode** (`false`): Filters at database level, only authorized documents are retrieved
   - **Notification mode** (`true`): Retrieves all documents, then separates accessible vs restricted and notifies user

## Configurable Pipelines

SMPTE-Copilot allows you to enable or disable individual pipeline steps through configuration. This provides flexibility to customize the processing flow based on your specific needs without modifying code.

### Overview

Both the **ingestion pipeline** and **query pipeline** support configurable steps that can be enabled or disabled via the `pipeline` section in `config.yaml`. When a step is disabled:
- The step is skipped during execution
- Associated components (loaders, embeddings, etc.) are not initialized, saving resources
- The pipeline continues with only the enabled steps

### Configuration

Configure pipeline steps in the `pipeline` section of `config.yaml`:

```yaml
pipeline:
  ingestion:
    load_enabled: true              
    preprocess_enabled: true        
    chunk_enabled: true             
    embedding_enabled: true         
    save_enabled: true
    # Parallelization settings (uses threading)
    parallel_enabled: false         # Enable parallel processing using threading
    max_workers: null               # Max parallel workers (null = CPU count, 1 = sequential)
  
  query:
    retrieve_enabled: true          
    rerank_enabled: false           # Enable reranking step (improves precision but adds latency)
    generation_enabled: true        
```

### Ingestion Pipeline Steps

The ingestion pipeline consists of four configurable steps:

1. **`load_enabled`** (Load Step)
   - Converts media files (PDF, images, videos, audio) to Markdown format
   - Required for: Processing source files
   - If disabled: You must provide pre-processed markdown files

2. **`preprocess_enabled`** (Preprocess Step)
   - Removes repeated headers, footers, and page numbers
   - Optional: Can be skipped if your documents don't have repetitions
   - If disabled: Raw text from loader is used directly

3. **`chunk_enabled`** (Chunk Step)
   - Splits the text into smaller, manageable chunks
   - Required for: Creating embeddings and storing in vector database
   - If disabled: Single large document will be used (not recommended)

4. **`save_enabled`** (Save Step)
   - Generates vector embeddings for each chunk and stores them in the vector database
   - This step includes embedding generation as they are dependent operations
   - Required for: Persisting data for later queries and enabling semantic search
   - If disabled: Processing happens but data is not saved (useful for testing)

**Note**: The save step includes embedding generation because embeddings are only useful when stored in the vector database. The embedding model and vector store are only created if `save_enabled` is true

### Parallel Ingestion

The ingestion pipeline supports parallel processing of multiple files to improve throughput when ingesting large batches of documents. Parallelization uses threading, which is ideal for the I/O-bound nature of document ingestion (file reading, API calls, database operations). This can significantly reduce total ingestion time, especially when processing many files.

**Configuration Parameters:**

1. **`parallel_enabled`** (default: `false`)
   - Enables parallel processing of files using threading
   - When `false`: Files are processed sequentially (one at a time)
   - When `true`: Multiple files are processed concurrently using thread pools

2. **`max_workers`** (default: `null`)
   - Maximum number of parallel workers
   - `null`: Uses the number of CPU cores available
   - `1`: Equivalent to sequential processing (parallel_enabled=false)
   - `> 1`: Specified number of parallel workers
   - Recommended: Start with `null` and adjust based on your system resources

**Example Configuration:**

```yaml
pipeline:
  ingestion:
    load_enabled: true
    preprocess_enabled: true
    chunk_enabled: true
    save_enabled: true
    parallel_enabled: true          # Enable parallel processing
    max_workers: 4                  # Use 4 parallel workers
```

**Considerations:**
- Ensure your vector store supports concurrent writes (most do)
- Monitor system resources when processing many large files
- API rate limits may affect parallel processing with external services
- Start with fewer workers and scale up as needed

### Query Pipeline Steps

The query pipeline consists of the following configurable steps:

1. **`retrieve_enabled`** (Retrieve Step)
   - Generates query embedding and retrieves relevant documents from the vector database
   - This step includes query embedding as they are dependent operations
   - Required for: Finding relevant context from your document corpus
   - If disabled: LLM will generate responses without document context

2. **`rerank_enabled`** (Rerank Step)
   - Reranks retrieved documents using a cross-encoder model for improved relevance
   - Optional: Improves precision but adds latency due to additional model inference
   - If disabled: Documents are ranked by vector similarity only

3. **`generation_enabled`** (Generation Step)
   - Uses an LLM to generate a natural language response based on retrieved documents
   - Optional: Can be disabled if you only want document retrieval
   - If disabled: Only retrieved documents are returned (no LLM response)

4. **Access Control Step** (automatic, based on `notify_on_denied_access`)
   - Separates accessible and restricted documents when `notify_on_denied_access: true`
   - Not a pipeline toggle - controlled by `access_control.notify_on_denied_access`
   - When enabled: Notifies users about restricted documents in the response

**Note**: The retrieve step includes query embedding because they are dependent operations - you can't retrieve documents without first embedding the query.

### Use Cases

#### Use Case 1: Testing Without Database Storage

Process and chunk documents without saving to the database:

```yaml
pipeline:
  ingestion:
    load_enabled: true
    preprocess_enabled: true
    chunk_enabled: true
    save_enabled: false            # Skip embedding and database storage
```

**Result**: Documents are processed and chunked, but embeddings are not generated and data is not saved to the database. Useful for testing preprocessing and chunking configurations.

#### Use Case 2: Skip Preprocessing

If your documents are clean and don't need preprocessing:

```yaml
pipeline:
  ingestion:
    load_enabled: true
    preprocess_enabled: false     # Skip preprocessing
    chunk_enabled: true
    embedding_enabled: true
    save_enabled: true
```

**Result**: Faster ingestion by skipping the preprocessing step.

#### Use Case 3: Document Retrieval Only

Retrieve relevant documents without LLM generation:

```yaml
pipeline:
  query:
    retrieve_enabled: true
    generation_enabled: false     # Skip LLM generation
```

**Result**: Returns retrieved documents with similarity scores, but no natural language response. Useful for building custom downstream processing or when you want to save LLM API costs.

#### Use Case 4: LLM Without Retrieval

Use LLM for direct question answering without document context:

```yaml
pipeline:
  query:
    retrieve_enabled: false       # Skip retrieval
    generation_enabled: true
```

**Result**: LLM generates responses without retrieving documents from the database. Useful for general knowledge questions or when you want the LLM to answer without specific context.

#### Use Case 5: High-Precision Retrieval with Reranking

Enable reranking to improve the quality of retrieved documents:

```yaml
pipeline:
  query:
    retrieve_enabled: true
    rerank_enabled: true          # Enable cross-encoder reranking
    generation_enabled: true

reranking:
  reranker_name: gemini
  reranker_config:
    model: gemini-2.5-flash
    api_key: ${GOOGLE_API_KEY}
```

**Result**: Retrieved documents are reranked using a cross-encoder model that better assesses query-document relevance. This improves precision at the cost of additional latency. Recommended for use cases where answer quality is more important than response time.

#### Use Case 6: Access Denial Notification

Notify users about restricted documents they cannot access:

```yaml
access_control:
  default_user_role: "Public"
  access_mapping_file: "./access_mapping.json"
  notify_on_denied_access: true   # Enable notification mode
```

**Result**: When users query documents, they receive their normal response plus a notification listing any additional relevant documents they lack permission to access. This helps users understand what information exists and what permissions they might need.

#### Use Case 7: High-Volume Parallel Ingestion

Process large batches of documents quickly using parallel processing:

```yaml
pipeline:
  ingestion:
    load_enabled: true
    preprocess_enabled: true
    chunk_enabled: true
    save_enabled: true
    parallel_enabled: true        # Enable parallel processing
    max_workers: null             # Use all available CPU cores
```

**Result**: Multiple files are processed concurrently using threading, significantly reducing total ingestion time. The executor automatically manages worker threads and provides progress tracking. Failed files are reported at the end without stopping the entire batch.

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
- Executes the ingestion pipeline: Load → Preprocess → Chunk → Embed → Save (see [Ingestion Pipeline](#ingestion-pipeline) for details)
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

## Evaluation


### Synthetic Testset Generation (Ragas)

Generate a synthetic evaluation dataset from the repository's markdown content using the `testgen` service.

```bash
# Generate a testset (adjust size as needed)
docker compose run --rm testgen \
  python evaluations/synthetic/generate_testset.py \
  --sources /app/data/markdown \
  --size 10 \
  --out /app/evaluations/synthetic/output/testset.jsonl

# Output will be written to evaluations/synthetic/output/testset.jsonl
```

- Note: The `testgen` service uses `evaluations/config.yaml`
