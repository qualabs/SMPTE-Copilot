# Dockerfile for RAG Ingestion & Retrieval Pipeline
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies and the package
# Using pyproject.toml as single source of truth
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Set Python path
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "--version"]

