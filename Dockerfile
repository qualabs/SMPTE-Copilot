# Multi-stage Dockerfile for RAG Pipeline with Docling, ChromaDB, and Whisper
FROM python:3.9-slim as base

# Metadata
LABEL maintainer="RAG Pipeline"
LABEL description="RAG Pipeline with Docling, ChromaDB, Whisper, and Gemini embeddings"

# Set working directory
WORKDIR /app

# Install system dependencies in a single layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build dependencies
    build-essential \
    g++ \
    # FFmpeg for video/audio processing
    ffmpeg \
    # OCR dependencies (for docling)
    tesseract-ocr \
    tesseract-ocr-eng \
    # Image processing libraries (updated for newer Debian)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional dependencies for docling
    poppler-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements file first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser files_parser.py .
COPY --chown=appuser:appuser audio_processor.py .

# Create directories for data with proper permissions
RUN mkdir -p /app/data \
    /app/chroma_db \
    /app/markdown_cache \
    /app/Transcriptions \
    /app/videos \
    /app/poc2_pdf && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Health check (optional - can be customized)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (can be overridden)
CMD ["python", "main.py"]

