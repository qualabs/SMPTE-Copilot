# RAG Pipeline with Docling, ChromaDB, and Whisper

A complete RAG (Retrieval-Augmented Generation) pipeline for processing PDF documents and video transcriptions, creating semantic embeddings, and enabling intelligent search.

## Features

- 📄 **PDF Processing**: Extract and process PDF documents using Docling with OCR and table detection
- 🎥 **Video Transcription**: Transcribe audio/video files using Whisper
- 🔍 **Semantic Chunking**: Intelligent document chunking using Docling's HybridChunker
- 🧠 **Vector Embeddings**: Generate embeddings using Google Gemini (3072 dimensions)
- 💾 **Vector Database**: Store and search using ChromaDB
- 🐳 **Dockerized**: Complete Docker setup for easy deployment

## Architecture

```
Video/PDF Files
    ↓
[Whisper Transcription] → JSON files
    ↓
[Docling DocumentConverter] → Structured Documents
    ↓
[HybridChunker] → Semantic Chunks (500 tokens max)
    ↓
[Gemini Embeddings] → 3072-dim Vectors
    ↓
[ChromaDB] → Vector Database
    ↓
[Semantic Search] → Query Results
```

## Prerequisites

- Docker and Docker Compose installed
- Google API Key for Gemini embeddings
- (Optional) Python 3.9+ and virtual environment for local development

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Test
```

### 2. Set Up Environment
```bash
# Create .env file with your Google API key
echo "API_KEY=your_google_api_key_here" > .env
```

### 3. Prepare Input Files

#### For Video/Audio Processing:
Place video files in `./videos/` directory, then run:
```bash
# Transcribe videos (creates JSON files)
docker-compose run --rm rag-pipeline python audio_processor.py
```

This creates JSON transcription files in `./Transcriptions/`

#### For PDF Processing:
Place PDF files in `./poc2_pdf/` directory

### 4. Run the Pipeline
```bash
# Build and run the complete pipeline
docker-compose up --build
```

This will:
1. Parse all JSON transcriptions (and PDFs if enabled)
2. Chunk documents semantically
3. Generate embeddings
4. Save to ChromaDB
5. Run a test query

## Project Structure

```
.
├── main.py                 # Main pipeline (ingestion + query)
├── files_parser.py         # Document parsing (PDF, DOCX, JSON)
├── audio_processor.py      # Video/audio transcription with Whisper
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── .gitignore            # Git ignore rules
├── README.md             # This file
│
├── Transcriptions/        # Input: JSON transcription files
├── videos/               # Input: Video files (for transcription)
├── poc2_pdf/             # Input: PDF documents
│
├── chroma_db/            # Output: ChromaDB database (auto-created)
└── markdown_cache/       # Output: Processed markdown files
```

## Configuration

### Modify Directories
Edit paths in `main.py` (lines 25-30):
```python
PDF_DIRECTORY = "/app/poc2_pdf"  # Docker path
TRANSCRIPTION_DIRECTORY = "/app/Transcriptions"  # Docker path
```

### Adjust Chunking
Edit `main.py` (line 72):
```python
chunker = HybridChunker(
    tokenizer=gemini_tokenizer,
    max_tokens=500,      # Max tokens per chunk
    merge_peers=False    # Don't merge similar chunks
)
```

### Change Query
Edit `main.py` (line 208):
```python
query = "Your custom question here"
```

## Usage Examples

### Process Only Transcriptions
```bash
docker-compose up
```

### Process Only PDFs
Uncomment PDF processing section in `main.py` (lines 87-93), then:
```bash
docker-compose up
```

### Run Individual Components

#### Transcribe Videos
```bash
docker-compose run --rm rag-pipeline python audio_processor.py
```

#### Parse Files Only
```bash
docker-compose run --rm rag-pipeline python files_parser.py
```

#### Query Only (After Ingestion)
Comment out ingestion sections in `main.py`, then:
```bash
docker-compose up
```

### Interactive Shell
```bash
docker-compose run --rm rag-pipeline /bin/bash
```

## Models Used

- **Whisper "turbo"**: Video/audio transcription
- **Gemini "embedding-001"**: Text embeddings (3072 dimensions)
- **Docling**: Document processing (OCR, layout, tables)
- **HybridChunker**: Semantic text chunking

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `API_KEY` | Google API key for Gemini embeddings | Yes |

## Output

### ChromaDB Collection
- **Name**: `test_collection_poc3`
- **Location**: `./chroma_db/`
- **Embedding Dimension**: 3072

### Chunk Metadata
Each chunk includes:
- `source_document`: Original filename
- `source_path`: Full file path
- `chunk_index`: Position in document
- `media_type`: "audio_transcript" or "pdf_document"
- `total_chunks_in_doc`: Total chunks in source document

## Troubleshooting

### No Files Found
- Check that JSON files exist in `./Transcriptions/`
- Verify paths in `main.py` match Docker volume mounts

### API Key Error
- Ensure `.env` file exists with `API_KEY=...`
- Check environment variable is loaded: `docker-compose config`





## Development

### Local Development (Without Docker)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variable
export API_KEY=your_key

# Run
python main.py
```



