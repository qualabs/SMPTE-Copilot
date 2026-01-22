# Configuration

This document provides a comprehensive guide to configuring SMPTE-Copilot through the `config.yaml` file.

## Table of Contents

- [Configuration Structure](#configuration-structure)
- [How Configuration Maps to Components](#how-configuration-maps-to-components)
- [Configuration Examples](#configuration-examples)
- [Access Control System](#access-control-system)
- [Configurable Pipelines](#configurable-pipelines)

## Configuration Structure

The `config.yaml` file is the central configuration file that controls which components are used and how they are configured. Each module has a corresponding section in the configuration file.

The `config.yaml` file is organized into sections that map to each module. See [`config-example.yaml`](../config-example.yaml) for a complete example with all available options and detailed comments.

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

## How Configuration Maps to Components

The configuration values directly map to the Enum types defined in each module:

- **`source_type`** → `InputSourceType` enum (e.g., `"S3"` -> `InputSourceType.S3`)
- **`loader.file_type_mapping`** → List of loader configurations. Each entry contains `extensions` (list of file extensions like `[.pdf, .docx]`), `loader_name` (e.g., `"pymupdf"` → `LoaderType.PYMUPDF`), and optional `loader_config`. Multiple extensions can share the same loader configuration to avoid repetition.
- **`preprocessing_name`** → `PreprocessorType` enum (e.g., `"rapidfuzz"` → `PreprocessorType.RAPIDFUZZ`)
- **`chunker_name`** → `ChunkerType` enum (e.g., `"langchain"` → `ChunkerType.LANGCHAIN`)
- **`embed_name`** → `EmbeddingModelType` enum (e.g., `"huggingface"` → `EmbeddingModelType.HUGGINGFACE`)
- **`store_name`** → `VectorStoreType` enum (e.g., `"chromadb"` -> `VectorStoreType.CHROMADB`, `"qdrant"` -> `VectorStoreType.QDRANT`)
- **`searcher_strategy`** → `RetrieverType` enum (e.g., `"similarity"` -> `RetrieverType.SIMILARITY`)
- **`reranker_name`** → `RerankerType` enum (e.g., `"gemini"` -> `RerankerType.GEMINI`)
- **`access_control`** → Access control configuration (see [Access Control System](#access-control-system) for details)
  - `notify_on_denied_access`: When `true`, enables notification mode that informs users about restricted documents

The system uses these values to:
1. Load the configuration from `config.yaml`
2. Map the string values to the corresponding Enum types
3. Use the Factory pattern to create instances of the selected components
4. Pass additional configuration parameters to the component constructors

**Note on Loader Configuration**: The `loader.file_type_mapping` allows you to configure different loaders for different file types. This enables the system to support multiple file formats (PDF, images, videos, audio) with appropriate loaders for each type. The format uses a list where each entry has an `extensions` list, allowing multiple extensions to share the same loader configuration. When adding support for a new file type, add the extension to an existing entry's `extensions` list (if it uses the same loader) or create a new entry.

## Configuration Examples

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
