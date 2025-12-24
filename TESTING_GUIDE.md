# Testing Guide for RAG Pipeline

## Prerequisites

1. **Environment Variables**
   ```bash
   # Create .env file with your Google API key
   echo "API_KEY=your_google_api_key_here" > .env
   ```

2. **Input Files**
   - JSON transcription files in `./Transcriptions/` directory
   - (Optional) PDF files in `./poc2_pdf/` directory

## Testing Methods

### Method 1: Using Docker Compose (Recommended)

#### Step 1: Build and Run
```bash
# Build the image and run the pipeline
docker-compose up --build
```

#### Step 2: Check Results
After completion, check:
- **ChromaDB**: `./chroma_db/` - Vector database files
- **Markdown Cache**: `./markdown_cache/` - Processed markdown files
- **Console Output**: Shows chunks created, embeddings generated, and query results

#### Step 3: View Logs
```bash
# If running in background
docker-compose logs -f rag-pipeline
```

### Method 2: Run Locally (Without Docker)

#### Step 1: Activate Virtual Environment
```bash
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

#### Step 2: Set Environment Variable
```bash
export API_KEY=your_google_api_key_here
```

#### Step 3: Run the Pipeline
```bash
python main.py
```

### Method 3: Test Individual Components

#### Test File Parser Only
```bash
# Using Docker
docker-compose run --rm rag-pipeline python files_parser.py

# Or locally
python files_parser.py
```

#### Test Audio Processor Only
```bash
# Using Docker
docker-compose run --rm rag-pipeline python audio_processor.py

# Or locally
python audio_processor.py
```

#### Test Query Only (After Ingestion)
Modify `main.py` to comment out ingestion and only run query section, then:
```bash
python main.py
```

## What to Verify

### 1. **File Parsing Phase**
Look for output like:
```
PHASE 1: PARSING FILES
📂 Scanning Transcription Directory: /Users/.../Transcriptions
Found X file(s) to process
Processing: filename.json
  ✓ Done
✅ Total Documents to Process: X
```

### 2. **Chunking & Embedding Phase**
Look for:
```
PHASE 2: CHUNKING & EMBEDDING
Processing File 1/X: filename.json
  -> Generated Y chunks
```

### 3. **Database Save Phase**
Look for:
```
PHASE 3: SAVING TO CHROMA DB
Deleted old collection.
✓ Saved X total chunks to database.
```

### 4. **Query Results**
Look for:
```
QUERY & RETRIEVAL
Query: How does a Rogue Leader attack...

Result #1 (Score: X.XXXX)
Source: filename.json
VIDEO TIMESTAMP DETECTED!  (if applicable)
--------------------------------------------------
[Chunk content here]
--------------------------------------------------
```

## Testing Different Queries

### Option 1: Modify main.py
Edit line 208 in `main.py`:
```python
query = "Your test query here"
```

### Option 2: Interactive Testing
Create a test script `test_query.py`:
```python
import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Setup
client = chromadb.PersistentClient(path="./chroma_db")
embeddings_func = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["API_KEY"],
    output_dimensionality=3072
)

vectorstore = Chroma(
    client=client,
    collection_name="test_collection_poc3",
    embedding_function=embeddings_func,
)

# Test query
query = input("Enter your query: ")
results = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nResult #{i} (Score: {score:.4f})")
    print(f"Source: {doc.metadata.get('source_document')}")
    print("-" * 50)
    print(doc.page_content[:500])  # First 500 chars
    print("-" * 50)
```

Run it:
```bash
python test_query.py
```

## Troubleshooting

### Issue: No files found
- Check that JSON files exist in `./Transcriptions/`
- Verify file paths in `main.py` (lines 21-22)

### Issue: API Key Error
- Verify `.env` file exists and contains `API_KEY=...`
- Check environment variable: `echo $API_KEY`

### Issue: Empty Results
- Verify chunks were created (check Phase 2 output)
- Check database was saved (check Phase 3 output)
- Try a more general query

### Issue: Collection Not Found
- The collection is created automatically
- If it doesn't exist, run the ingestion phase first

## Quick Test Checklist

- [ ] `.env` file created with API_KEY
- [ ] JSON files in `./Transcriptions/` directory
- [ ] Docker image built successfully
- [ ] Pipeline runs without errors
- [ ] Chunks are generated (check Phase 2 output)
- [ ] Database saves successfully (check Phase 3 output)
- [ ] Query returns results (check Phase 4 output)
- [ ] Results show source documents and content





