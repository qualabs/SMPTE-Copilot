# Dockerfile for RAG Ingestion & Retrieval Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "--version"]

