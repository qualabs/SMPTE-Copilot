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

For detailed instructions on enabling clickable citations and advanced OpenWebUI features, see the [Advanced Usage](docs/usage.md#openwebui-integration) documentation.

## Project Structure

The project is organized into modular components that follow a consistent pattern. Each module implements the Factory pattern to enable easy extension and addition of new components. For detailed information about the architecture, see the [Architecture](docs/architecture.md#module-architecture) documentation.

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

## Documentation

For detailed information about different aspects of the project, see:

- **[Architecture](docs/architecture.md)** - Architectural patterns, Factory Pattern, Pipeline Pattern, and module structure
- **[Configuration](docs/configuration.md)** - Complete guide to `config.yaml`, Access Control, and configurable pipelines
- **[Development Guide](docs/development.md)** - How to add new components and extend the system
- **[Advanced Usage](docs/usage.md)** - Detailed CLI, use cases, and advanced features

## Quick Links

- **Getting Started**: See [Execution](#execution) above for basic setup
- **Configuration**: See [`config-example.yaml`](config-example.yaml) for a complete configuration example
- **Adding Components**: See [Development Guide](docs/development.md) for step-by-step instructions
- **Pipeline Details**: See [Architecture](docs/architecture.md#pipeline-pattern-architecture) for pipeline flow and extensibility
