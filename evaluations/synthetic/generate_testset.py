import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from ragas.testset.synthesizers.generate import TestsetGenerator
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import yaml


# Defaults
DEFAULT_DATA_DIR = "data/markdown"
DEFAULT_LLM_MODEL = "gemini-1.5-pro"
DEFAULT_EMBED_MODEL = "text-embedding-004"


def main() -> int:
    """Entry point: generate a synthetic testset from markdown corpus."""
    parser = argparse.ArgumentParser(description="Generate synthetic RAG testset using Ragas")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=[DEFAULT_DATA_DIR],
        help="Source files/folders to read text from",
    )
    parser.add_argument("--size", type=int, default=10, help="Number of samples to generate")
    parser.add_argument(
        "--out",
        type=str,
        default="evaluations/synthetic/output/testset.jsonl",
        help="Output file path (JSONL)",
    )
    args = parser.parse_args()

    # Load configuration from root YAML (mounted into container)
    cfg = load_yaml_config("config.yaml")

    # Read corpus
    corpus = load_corpus(args.sources)
    if not corpus:
        print("No source texts found in provided paths.")
        return 1

    # Resolve models and credentials
    google_api_key = resolve_google_api_key(cfg)
    if not google_api_key:
        print("Google API key not found. Set it in config.yaml under llm.api_key or embedding.google_api_key.")
        return 1

    llm_model, embed_model = resolve_models(cfg)

    # Instantiate adapters
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0, google_api_key=google_api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=google_api_key)

    # Generate testset
    generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    testset = generator.generate_with_chunks(chunks=corpus, testset_size=args.size)

    # Save output
    write_jsonl(testset.to_list(), args.out)
    print(f"Saved {len(testset.samples)} samples to {args.out}")
    return 0


# Helpers
def resolve_google_api_key(cfg: dict) -> Optional[str]:
    """Resolve Google API key strictly from root config (no env).

    Tries `llm.llm_config.api_key` then `embedding.embed_config.google_api_key`.
    """
    llm_cfg = (cfg.get("llm") or {}).get("llm_config") or {}
    if isinstance(llm_cfg, dict) and llm_cfg.get("api_key"):
        return llm_cfg.get("api_key")

    emb_cfg = (cfg.get("embedding") or {}).get("embed_config") or {}
    if isinstance(emb_cfg, dict) and emb_cfg.get("google_api_key"):
        return emb_cfg.get("google_api_key")

    return None


def resolve_models(cfg: dict) -> Tuple[str, str]:
    """Resolve model names with env overrides and config fallbacks."""
    llm_model = os.environ.get("RAGAS_LLM_MODEL") or (
        (cfg.get("llm") or {}).get("llm_config", {}).get("model")
    ) or DEFAULT_LLM_MODEL

    embed_model = os.environ.get("RAGAS_EMBED_MODEL") or (
        (cfg.get("embedding") or {}).get("embed_config", {}).get("model_name")
    ) or DEFAULT_EMBED_MODEL

    return llm_model, embed_model


def load_yaml_config(path: str) -> dict:
    """Load YAML config as a dict, returning empty dict on error."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_corpus(paths: List[str]) -> List[str]:
    """Load text content from given files or directories (md/txt)."""
    texts: List[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in (".md", ".txt"):
                for fp in path.rglob(f"*{ext}"):
                    try:
                        content = fp.read_text(encoding="utf-8", errors="ignore")
                        if content.strip():
                            texts.append(content)
                    except Exception:
                        continue
        elif path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                if content.strip():
                    texts.append(content)
            except Exception:
                continue
    return texts


def write_jsonl(data: List[dict], out_path: str) -> None:
    """Write list of dicts to JSONL file."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
