# Advanced Usage

This document covers advanced usage scenarios, CLI details, and integration guides for SMPTE-Copilot.

## Table of Contents

- [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [ingest.py - Document Ingestion](#ingestpy---document-ingestion)
  - [query.py - Document Querying](#querypy---document-querying)
- [OpenWebUI Integration](#openwebui-integration)
  - [Enabling Clickable Citations](#enabling-clickable-citations)

## Command-Line Interface (CLI)

The project provides two main CLI commands for ingesting documents and querying the vector database.

### `ingest.py` - Document Ingestion

The `ingest.py` command processes media files and adds them to the vector database using the [ingestion pipeline](architecture.md#ingestion-pipeline). It is designed to support multiple file types including PDFs, images, videos, and audio files (currently supports PDFs, with multimodal support planned).

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
- Executes the ingestion pipeline: Load → Preprocess → Chunk → Embed → Save (see [Ingestion Pipeline](architecture.md#ingestion-pipeline) for details)
- Saves processed content (e.g., Markdown files) to the configured output directory (`paths.markdown_dir`)
- Provides detailed logging of each step in the pipeline

**Output:**
- Processed content files saved to the configured output directory
- Vector database populated with embedded document chunks
- Summary of processed files, chunk counts, and database location

### `query.py` - Document Querying

The `query.py` command searches the vector database for documents similar to a given query using the [query pipeline](architecture.md#query-pipeline).

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
3. Executes the query pipeline: QueryEmbedding → Retrieve (see [Query Pipeline](architecture.md#query-pipeline) for details)
4. Displays results with similarity scores and document content

**Output:**
- List of relevant documents ranked by similarity score
- Each result includes:
  - Similarity score (higher = more similar)
  - Document content (chunk text)
  - Metadata (source file, page numbers, etc.)

**Important**: The embedding model used for querying must match the one used during ingestion to ensure accurate similarity search.

## OpenWebUI Integration

SMPTE-Copilot can be used with **OpenWebUI** as a chat interface via its OpenAI-compatible API. Once the `api` and `openwebui` services are running, access the UI at **http://localhost:3000**. The backend is automatically configured through `OPENAI_API_BASE_URL` to use the local RAG API (`/v1/chat/completions`). 

To start OpenWebUI with SMPTE-Copilot:

```bash
docker compose up openwebui
```

### Enabling Clickable Citations

By default, the integration uses the standard OpenAI-compatible endpoint. To enable **clickable citations** that show the source chunks when clicked, you need to install the SMPTE Copilot Pipe in OpenWebUI:

1. **Access Admin Settings**: Log into OpenWebUI as an admin and go to **Admin Panel** → **Settings** → **Functions**

2. **Create New Function**: Click the **+** button to create a new function

3. **Copy Pipe Code**: Copy the entire contents of [`src/openwebui/smpte_pipe.py`](../src/openwebui/smpte_pipe.py) and paste it into the function editor

4. **Save and Enable**: Save the function and ensure it's enabled

5. **Configure Valves** (if needed): Click the gear icon on the function to configure:
   - `SMPTE_API_BASE_URL`: The backend URL (default: `http://api:8000`)
   - `REQUEST_TIMEOUT`: Request timeout in seconds (default: `120`)

6. **Use the Pipe**: In your chat, select the model **"SMPTE Copilot RAG"** instead of the standard "smpte-copilot" model

When using the Pipe, citations like `[1]`, `[2]`, etc. in the response will be clickable, showing a popup with the source document name, page number, and the actual chunk content that was retrieved.

**Note**: The Pipe uses the `/v1/rag/query` endpoint which returns both the response and citation metadata, while the standard OpenAI-compatible endpoint (`/v1/chat/completions`) only returns the response text.

For more information about configuration options, see the [Configuration](configuration.md) documentation. For details on access control and pipeline configuration, see the [Configuration Guide](configuration.md).
