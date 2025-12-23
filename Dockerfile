# Dockerfile for SMPTE-Copilot (CLI tools and API server)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies
# Using pyproject.toml as single source of truth
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "--version"]

