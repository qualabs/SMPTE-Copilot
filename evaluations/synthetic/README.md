## Synthetic Test Data (Ragas)

Generate synthetic query/context/reference samples from your repo content.

### Quick Run

```
# Using Docker (recommended)
docker compose build
docker compose run --rm testgen python evaluations/synthetic/generate_testset.py \
  --sources /app/data/markdown \
  --size 10 \
  --out /app/evaluations/synthetic/output/testset.jsonl
```


### Defaults and Overrides
- Source: `data/markdown` (recursively loads `.md`/`.txt`).
- Models: `gemini-1.5-pro` (LLM), `text-embedding-004` (embeddings).
- Override via env: `RAGAS_LLM_MODEL`, `RAGAS_EMBED_MODEL`.
- Config: uses `evaluations/config.yaml` mounted to `/app/config.yaml`.
- Key resolution: strictly from that config → `llm.api_key` or `embedding.google_api_key`.

### Output
- Writes JSONL at the specified `--out`, compatible with `Testset.to_list()`.

### Note
- Synthetic generation uses LLMs and embeddings; size impacts cost.