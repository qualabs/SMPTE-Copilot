"""
Simple RAG Evaluation Script
Minimal evaluation: Question → RAG API → Compare with ground truth
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    AnswerRelevancy,
)
from ragas.embeddings import LangchainEmbeddingsWrapper


# Defaults 
DEFAULT_CONFIG = "evaluations/config.yaml"
DEFAULT_TESTSET = "evaluations/synthetic/output/testset.jsonl"
DEFAULT_OUTPUT_DIR = "evaluations/metrics/output"

DEFAULT_API_TIMEOUT_S = 60.0
SEPARATOR_WIDTH = 50

# Metric columns to display (and average)
METRIC_COLUMNS: Sequence[Tuple[str, str]] = (
    ("llm_context_precision_with_reference", "Context Precision"),
    ("context_recall", "Context Recall"),
    ("faithfulness", "Faithfulness"),
    ("answer_correctness", "Answer Correctness"),
    ("answer_relevancy", "Answer Relevancy"),
)
"""
RAGAS METRICS DEFINITIONS:
- Context Precision: (Signal-to-Noise) Are the retrieved documents actually useful, or is there too much noise?
- Context Recall:    (Coverage) Did we find *all* the necessary information needed to answer?
- Faithfulness:      (Anti-Hallucination) Is the answer based *only* on the provided context, without making things up?
- Answer Correctness:(Accuracy) Does the answer match the expert reference (Ground Truth)?
- Answer Relevancy:  (User Relevance) Does the answer directly address the user's question without being vague?
"""

def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    api_url = config["api"]["base_url"]
    timeout_s = float(config.get("api", {}).get("timeout", DEFAULT_API_TIMEOUT_S))

    print(f"Loading testset: {args.testset}")
    testset = load_testset(args.testset)
    print(f"Found {len(testset)} questions\n")

    print("=" * SEPARATOR_WIDTH)
    print("STEP 1: Getting answers from RAG API")
    print("=" * SEPARATOR_WIDTH)
    results = fetch_rag_responses(api_url=api_url, testset=testset, timeout_s=timeout_s)

    if not results:
        print("\nNo successful responses. Check if RAG API is running.")
        return 1

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("STEP 2: Evaluating with RAGAS")
    print("=" * SEPARATOR_WIDTH)

    llm, embeddings = setup_ragas_models(config)
    eval_results = run_evaluation(results=results, llm=llm, embeddings=embeddings)

    print("\n" + "=" * SEPARATOR_WIDTH)
    print("RESULTS")
    print("=" * SEPARATOR_WIDTH)
    df = eval_results.to_pandas()
    print_results(df)

    output_dir = Path(args.output)
    save_results(df=df, output_dir=output_dir)
    print(f"\nResults saved to {output_dir / 'results.csv'}")
    print(f"Averages saved to {output_dir / 'results.json'}")
    return 0


def fetch_rag_responses(
    *,
    api_url: str,
    testset: Sequence[Dict[str, Any]],
    timeout_s: float,
) -> List[Dict[str, Any]]:
    """Call RAG API for each test question and return rows."""
    results: List[Dict[str, Any]] = []
    total = len(testset)

    for i, item in enumerate(testset, 1):
        question = str(item.get("user_input", ""))
        ground_truth = str(item.get("reference", ""))

        print(f"[{i}/{total}] {question[:60]}...")
        response, contexts = call_rag_api(api_url, question, timeout_s=timeout_s)

        if response:
            print(f"  ✓ Got answer ({len(response)} chars, {len(contexts)} contexts)")
            results.append(
                {
                    "user_input": question,
                    "response": response,
                    "retrieved_contexts": contexts,
                    "reference": ground_truth,
                }
            )
        else:
            print("  ✗ Failed - skipping")

    return results


def setup_ragas_models(config: Dict[str, Any]):
    """Initialize LLM + embeddings for RAGAS."""
    api_key = config["llm"]["llm_config"]["api_key"]
    llm_model = config["llm"]["llm_config"]["model"]
    embed_model = config["embedding"]["embed_config"]["model_name"]

    # Configure Gemini to return raw JSON (no markdown code blocks)
    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=api_key,
        temperature=0,
        model_kwargs={"response_mime_type": "application/json"},
    )

    # Wrap LangChain embeddings for RAGAS compatibility
    langchain_embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embed_model}",
        google_api_key=api_key,
    )
    embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    return llm, embeddings


def run_evaluation(*, results: Sequence[Dict[str, Any]], llm, embeddings):
    """Run RAGAS evaluation on the collected responses."""
    dataset = EvaluationDataset.from_list(list(results))
    metrics = [
        LLMContextPrecisionWithReference(llm=llm),  # Are retrieved contexts relevant?
        LLMContextRecall(llm=llm),  # Does context contain all needed info?
        Faithfulness(llm=llm),  # Is answer grounded in context?
        AnswerCorrectness(llm=llm, embeddings=embeddings),  # Does answer match expected?
        AnswerRelevancy(llm=llm, embeddings=embeddings),  # Is answer relevant to the question?
    ]

    print(f"Running {len(metrics)} metrics on {len(results)} samples...")
    print("(This may take a few minutes)\n")
    return evaluate(dataset=dataset, metrics=metrics)


def print_results(df: pd.DataFrame) -> None:
    """Print per-sample and average metrics."""
    for i, row in df.iterrows():
        print(f"\nSample {i + 1}: {str(row.get('user_input', ''))[:50]}...")
        for col, label in METRIC_COLUMNS:
            if col in df.columns and pd.notna(row.get(col)):
                print(f"  {label:20s}: {row[col]:.2f}")
            else:
                print(f"  {label:20s}: N/A")

    print("\n" + "-" * SEPARATOR_WIDTH)
    print("AVERAGES")
    print("-" * SEPARATOR_WIDTH)
    for col, label in METRIC_COLUMNS:
        if col in df.columns:
            avg = df[col].dropna().mean()
            if pd.notna(avg):
                print(f"  {label:20s}: {avg:.2f}")
            else:
                print(f"  {label:20s}: N/A")


def save_results(*, df: pd.DataFrame, output_dir: Path) -> Path:
    """Save detailed results to CSV and averages to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    
    # Calculate averages (same as printed)
    averages = {}
    for col, label in METRIC_COLUMNS:
        if col in df.columns:
            avg = df[col].dropna().mean()
            if pd.notna(avg):
                averages[label] = round(float(avg), 2)
            else:
                averages[label] = None
    
    # Save JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(averages, f, indent=2)
    
    return csv_path


def call_rag_api(base_url: str, question: str, *, timeout_s: float) -> Tuple[Optional[str], List[str]]:
    """Call RAG API and return (response, contexts)."""
    try:
        r = requests.post(
            f"{base_url}/v1/rag/query",
            json={"query": question},
            headers={"Content-Type": "application/json"},
            timeout=timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        response = data.get("response", "")
        contexts = [c.get("content", "") for c in data.get("citations", [])]
        return response, contexts
    except Exception as e:
        print(f"  Error: {e}")
        return None, []


def load_testset(path: str) -> List[Dict[str, Any]]:
    """Load testset from JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple RAG Evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config file")
    parser.add_argument("--testset", default=DEFAULT_TESTSET, help="Testset file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
