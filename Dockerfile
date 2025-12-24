# Dockerfile for RAG Ingestion & Retrieval Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Build dependencies and libraries for Docling document processing
# - build-essential, g++: For compiling Python packages
# - libgl1, libglib2.0-0: For OpenCV (headless mode) - avoids libGL/libgthread errors
# - libsm6, libxext6, libxrender-dev, libgomp1: Additional image processing libraries for better compatibility
# - poppler-utils: PDF processing utilities (required for Docling PDF parsing)
# - tesseract-ocr, tesseract-ocr-eng: OCR engine for text extraction from images (enables better table detection)
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies and the package
# Using pyproject.toml as single source of truth
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[gemini,docling]"

# Set Python path and OpenCV headless mode (for OCR and table detection)
ENV PYTHONPATH=/app
ENV OPENCV_HEADLESS=1

# Default command (can be overridden)
CMD ["python", "--version"]

