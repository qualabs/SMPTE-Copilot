# Dockerfile for evaluations (synthetic testset generation and scoring)
FROM python:3.12-slim

WORKDIR /app

# Minimal system setup (kept lean for evaluations)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy evaluation code and dependencies config (context is evaluations/)
COPY . /app/evaluations/

# Install Python dependencies defined for evaluations
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e /app/evaluations

ENV PYTHONPATH=/app

CMD ["python", "evaluations/synthetic/generate_testset.py", "--help"]
