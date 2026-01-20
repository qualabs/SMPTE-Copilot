# Evaluations

Run synthetic testset generation and RAG evaluation fully on Docker using the single config at `evaluations/config.yaml`.

## Synthetic Testset

Generate queries, contexts, and references from your markdown corpus.

```bash
# Build eval image
docker compose build testgen

# Generate testset (defaults read from /app/config.yaml)
docker compose run --rm testgen \
  python evaluations/synthetic/generate_testset.py \
  --sources /app/data/markdown \
  --size 10 \
  --out /app/evaluations/synthetic/output/testset.jsonl
```

- Config: [evaluations/config.yaml](evaluations/config.yaml) is mounted to `/app/config.yaml`.
- Output: [evaluations/synthetic/output/testset.jsonl](evaluations/synthetic/output/testset.jsonl)

## RAG Evaluation

Compute Ragas metrics over the generated testset using Gemini and your persisted vector store (Qdrant by default).

```bash
# Ensure Qdrant is up (if using persisted store)
docker compose up -d qdrant

# Build eval image (if not already)
docker compose build ragscore

# Run evaluation and save metrics
docker compose run --rm ragscore \
  python evaluations/metrics/run_eval.py \
  --testset /app/evaluations/synthetic/output/testset.jsonl \
  --out-json /app/evaluations/metrics/output/results.json \
  --out-csv /app/evaluations/metrics/output/detailed.csv \
  --k 5
```

- Outputs: 
  - Summary JSON: [evaluations/metrics/output/results.json](evaluations/metrics/output/results.json)
  - Per-sample CSV: [evaluations/metrics/output/detailed.csv](evaluations/metrics/output/detailed.csv)

## Configuration

- Single config: [evaluations/config.yaml](evaluations/config.yaml) provides Gemini credentials, model names, paths, and vector store settings.
- Keys: Gemini API key is read from `llm.llm_config.api_key` or `embedding.embed_config.google_api_key`.
- Vector store: defaults to Qdrant (`qdrant` service); local Chroma fallback is used if the store is unavailable.

## Troubleshooting

- If you see Qdrant client/server version warnings, the evaluation runner disables strict compatibility checks by default. You can pin `qdrant-client` to match the server if desired.
- Long runs: Evaluations invoke the LLM for multiple metrics; expect some latency.
- For quick sanity checks, reduce `--size` in testset generation and `--k` in evaluation.
