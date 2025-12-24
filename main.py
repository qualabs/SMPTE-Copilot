from pathlib import Path
from google import genai
from google.genai import types
from pydantic import ConfigDict
import os
import chromadb
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from typing import List, Optional, Any

# IMPORT YOUR NEW PARSER
from files_parser import parse_directory

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Define your directories
# Use container paths (Docker) or absolute paths (local)
# For Docker: use /app/... paths
# For local: use absolute paths
import os
if os.path.exists("/app/Transcriptions"):  # Docker container
    PDF_DIRECTORY = "/app/poc2_pdf"
    TRANSCRIPTION_DIRECTORY = "/app/Transcriptions"
else:  # Local development
    # Use relative paths from project root
    project_root = Path(__file__).parent
    PDF_DIRECTORY = str(project_root / "poc2_pdf")
    TRANSCRIPTION_DIRECTORY = str(project_root / "Transcriptions")

# Classes (Tokenizer & Embeddings)
class GeminiTokenizer(BaseTokenizer):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: genai.Client
    model: str = "gemini-embedding-001"
    max_tokens: int = 2048
    
    def count_tokens(self, text: str) -> int:
        try:
            response = self.client.models.count_tokens(model=self.model, contents=text)
            return response.total_tokens
        except Exception:
            return len(text) // 4
    
    def get_max_tokens(self) -> int:
        return self.max_tokens
    
    def get_tokenizer(self) -> Any:
        return self.client

class CustomGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    output_dimensionality: Optional[int] = None
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text, output_dimensionality=self.output_dimensionality)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts, output_dimensionality=self.output_dimensionality)

# ==========================================
# 2. INGESTION & PROCESSING - COMMENTED OUT (Only querying)
# ==========================================

# Initialize Clients
embedding_client = genai.Client(api_key=os.environ["API_KEY"])

gemini_tokenizer = GeminiTokenizer(
    client=embedding_client,
    model="gemini-embedding-001"
)

# Initialize HybridChunker (Same logic for Audio and PDF!)
chunker = HybridChunker(tokenizer=gemini_tokenizer,max_tokens=500,merge_peers=False)

# Storage lists
my_chunks = []
my_vectors = []
my_ids = []
my_metadata = []

# --- A. PARSE ALL FILES (PDFs + TRANSCRIPTS) ---
print(f"\n{'='*60}")
print("PHASE 1: PARSING FILES")
print(f"{'='*60}")

all_docling_results = []

# 1. Parse PDFs (COMMENTED OUT - Only processing audio/transcriptions)
# print(f"📂 Scanning PDF Directory: {PDF_DIRECTORY}")
# if Path(PDF_DIRECTORY).exists():
#     pdf_results = parse_directory(PDF_DIRECTORY, output_dir="markdown_cache")
#     all_docling_results.extend(pdf_results)
# else:
#     print("⚠️ PDF Directory not found.")

# 2. Parse Transcripts (Audio/Video)
print(f"📂 Scanning Transcription Directory: {TRANSCRIPTION_DIRECTORY}")
if Path(TRANSCRIPTION_DIRECTORY).exists():
    # parse_directory automatically handles .json and .txt conversion
    trans_results = parse_directory(TRANSCRIPTION_DIRECTORY, output_dir="markdown_cache")
    all_docling_results.extend(trans_results)
else:
    print("⚠️ Transcription Directory not found.")

print(f"\n✅ Total Documents to Process: {len(all_docling_results)}")


# --- B. CHUNK & EMBED ---
print(f"\n{'='*60}")
print("PHASE 2: CHUNKING & EMBEDDING")
print(f"{'='*60}")

for i, (conversion_result, md_content, output_path) in enumerate(all_docling_results):
    # Extract the Docling Document object
    # Note: conversion_result is the object returned by converter.convert()
    doc = conversion_result.document
    original_filename = conversion_result.input.file.name
    file_path_str = str(conversion_result.input.file)
    
    print(f"\nProcessing File {i+1}/{len(all_docling_results)}: {original_filename}")
    
    # CHUNKING
    chunk_iter = chunker.chunk(dl_doc=doc)
    
    chunk_counter = 0
    file_chunks_start_idx = len(my_chunks)
    
    for chunk in chunk_iter:
        # Generate Embedding
        try:
            embedding_result = embedding_client.models.embed_content(
                model="gemini-embedding-001", 
                contents=chunk.text,
                config=types.EmbedContentConfig(output_dimensionality=3072),
            )
            vector = embedding_result.embeddings[0].values
            
            # Metadata
            # We add a 'media_type' inference based on extension for filtering later if needed
            is_transcript = original_filename.endswith(('.json', '.txt'))
            
            meta = {
                "source_document": original_filename,
                "source_path": file_path_str,
                "chunk_index": chunk_counter,
                "media_type": "audio_transcript" if is_transcript else "pdf_document",
                "total_chunks_in_doc": None 
            }
            
            my_chunks.append(chunk.text)
            my_vectors.append(vector)
            my_ids.append(f"{original_filename}_chunk_{chunk_counter}")
            my_metadata.append(meta)
            
            chunk_counter += 1
            
        except Exception as e:
            print(f"⚠️ Error embedding chunk {chunk_counter}: {e}")

    # Update total counts for this file's chunks
    for j in range(chunk_counter):
        my_metadata[file_chunks_start_idx + j]["total_chunks_in_doc"] = chunk_counter
        
    print(f"  -> Generated {chunk_counter} chunks")

# ==========================================
# 3. SAVE TO DATABASE
# ==========================================
print(f"\n{'='*60}")
print("PHASE 3: SAVING TO CHROMA DB")
print(f"{'='*60}")

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection(name="test_collection_poc3")
    print("Deleted old collection.")
except Exception:
    pass

collection = client.get_or_create_collection(name="test_collection_poc3")

if my_chunks:
    collection.add(
        documents=my_chunks,
        ids=my_ids,
        embeddings=my_vectors,
        metadatas=my_metadata,
    )
    print(f"✓ Saved {len(my_chunks)} total chunks to database.")
else:
    print("⚠️ No chunks to save.")


# ==========================================
# QUERY & RETRIEVAL (ONLY ACTIVE SECTION)
# ==========================================
print(f"\n{'='*60}")
print("QUERY & RETRIEVAL")
print(f"{'='*60}")

# Initialize ChromaDB client (needed for querying)
client = chromadb.PersistentClient(path="./chroma_db")


embeddings_func = CustomGoogleEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["API_KEY"],
    output_dimensionality=3072
)

vectorstore = Chroma(
    client=client,
    collection_name="test_collection_poc3",
    embedding_function=embeddings_func,
)

query = "How does a Rogue Leader attack exploit the Best Master Clock Algorithm (BMCA) in PTP networks?"
print(f"Query: {query}\n")

results = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nResult #{i} (Score: {score:.4f})")
    print(f"Source: {doc.metadata.get('source_document')}")
    
    if "**(Time:" in doc.page_content:
        print("VIDEO TIMESTAMP DETECTED!")
        
    print("-" * 50)
    
    print(f"{doc.page_content}") 
    print("-" * 50)
