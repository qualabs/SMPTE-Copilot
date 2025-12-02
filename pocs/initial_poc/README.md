# Initial POC

Quick test guide for the RAG Ingestion & Retrieval Pipeline.

## About This Project

This is a **Proof of Concept (POC)** that serves as an initial exploration of RAG (Retrieval-Augmented Generation) technology. The main objectives are to:

- Test basic pipeline components and technologies
- Assess the **feasibility** of implementing a RAG solution
- Evaluate **ease of setup and deployment** using containerized environments
- Establish a foundation for future development and scaling

### What is Ingestion?

**Ingestion** is the process of preparing documents for search.

1. **Documents are converted** from PDF format into a searchable format (Markdown)
2. **Text is broken into chunks** - smaller, manageable pieces that preserve context
3. **Each chunk is converted into a embedding**. Is a mathematical representation that captures the meaning
4. **Embeddings are stored** in a database optimized for similarity search

Once ingested, documents are ready to be searched and retrieved based on meaning.

### What is Querying?

**Querying** is the process of finding relevant information from ingested documents:

1. **Your question is converted** into the same type of embedding used during ingestion
2. **The system searches** the database for document chunks with similar embedding
3. **Results are ranked** by how closely they match your question
4. **Relevant chunks are returned** with similarity scores showing how well they match

This allows you to ask questions in natural language and get back the most relevant parts of your documents, even if they don't contain the exact words you used.

---

## Prerequisites

- Docker and Docker Compose installed

**Check if Docker is already installed:**
```bash
docker --version
docker-compose --version
```

---

## Test Steps

### 1. Clone the Branch

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Or checkout the branch if you already have the repo
git checkout <branch-name>
```

You should now be in the project root directory (where `docker-compose.yml` is located).

### 2. Verify Test Data

```bash
# Check that the sample PDF is in the data folder
ls -la data/
```

**Note:** The project includes a sample PDF in the `data/` folder for testing.

### 3. Build Docker Image

```bash
# From the project root directory
docker-compose build
```

### 4. Ingest Document

```bash
# Option 1: Ingest a specific PDF
docker-compose run --rm ingest python ingest.py /app/data/Sample-pdf.pdf

# Option 2: Ingest all PDFs in the data folder
docker-compose run --rm ingest python ingest.py /app/data/
```

**Expected:** `✓ Ingestion Complete!`

### 5. Query Database

```bash
docker-compose run --rm query python query.py "What is this document about?"
```

**Expected:** Results with similarity scores

---

## Directory Structure

After setup, your project should look like:

```
<project-root>/
├── data/              ← Sample PDF included here
│   └── Sample-pdf.pdf ← Test file (already included)
├── chroma_db/         ← Auto-created after ingestion
├── docker-compose.yml
├── Dockerfile
├── ingest.py
├── query.py
└── src/
```

---

## Verification

- [ ] Ingestion completes without errors
- [ ] `chroma_db/` directory is created in project root
- [ ] Query returns results with scores

---

## Troubleshooting

**"Cannot connect to Docker daemon"**
```bash
open -a Docker  # macOS
```

**"No such file or directory: /app/data/Sample-pdf.pdf"**
```bash
# Make sure you're in the project root
pwd  # Should show the directory with docker-compose.yml

# Verify PDF is in ./data/
ls -la data/
```

**"Query returned 0 results"**
```bash
# Run ingestion first
docker-compose run --rm ingest python ingest.py /app/data/Sample-pdf.pdf
```

**"ModuleNotFoundError"**
```bash
docker-compose build --no-cache
```

---

## Quick Commands

```bash
# Build
docker-compose build

# Ingest a specific PDF
docker-compose run --rm ingest python ingest.py /app/data/Sample-pdf.pdf

# Ingest all PDFs in data folder
docker-compose run --rm ingest python ingest.py /app/data/

# Query
docker-compose run --rm query python query.py "your question"

# Clean up
docker-compose down
rm -rf chroma_db/
```

---
