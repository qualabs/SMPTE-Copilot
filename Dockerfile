# Dockerfile for RAG Ingestion & Retrieval Pipeline
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    g++ \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (for better Docker layer caching)
COPY pyproject.toml .

# Install Python dependencies from pyproject.toml BEFORE copying source code
# This way, dependencies are only reinstalled when pyproject.toml changes,
# not when source code changes.
#
# pip install -e . needs src/ to exist, so we create a minimal one temporarily.
# After installing dependencies + package, we remove only the package,
# keeping all dependencies installed.
RUN mkdir -p src && \
    touch src/__init__.py && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    pip uninstall -y rag-ingestion && \
    rm -rf src

# Copy source code (this layer only invalidates when source changes)
COPY src/ ./src/

# Install the package in editable mode (fast: dependencies already installed)
RUN pip install --no-cache-dir -e . --no-deps

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "--version"]