# SMPTE-Copilot
An open-source AI co-pilot that ingests and indexes text, audio, and video to enable semantic, multimodal search of media archives. The prototype provides modular ingestion, a chat-based retrieval pipeline, transparent citations, and tiered access for public users, members, and staff.

# Execution code

```bash
# Build
docker-compose build

# Ingest all PDFs in data folder
docker-compose run --rm ingest python src/cli/ingest.py /app/data/

# Query
docker-compose run --rm query python src/cli/query.py "your question"

# Clean up
docker-compose down
```