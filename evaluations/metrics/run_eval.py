"""
RAG Evaluation Script
Evaluates a RAG pipeline using RAGAS metrics against a test dataset.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import requests
import yaml
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.embeddings import LangchainEmbeddingsWrapper

# Configuration defaults
DEFAULT_CONFIG = "evaluations/config.yaml"
DEFAULT_TESTSET = "evaluations/synthetic/output/testset.jsonl"
DEFAULT_RESULTS_JSON = "evaluations/metrics/output/results.json"
DEFAULT_RESULTS_CSV = "evaluations/metrics/output/detailed.csv"


def main() -> int:
    """Main entry point for RAG evaluation."""
    args = parse_arguments()
    config = load_config(args.config)

    # Get settings from config
    api_base_url = config["api"]["base_url"]
    timeout = config.get("api", {}).get("timeout", 30.0)
    api_key = config["llm"]["llm_config"]["api_key"]
    llm_model = config["llm"]["llm_config"]["model"]
    embed_model = config["embedding"]["embed_config"]["model_name"]

    # Load testset
    print(f"Loading testset: {args.testset}")
    test_items = load_testset(args.testset)
    print(f"Found {len(test_items)} questions\n")

    # Step 1: Get answers from RAG API
    print("=" * 60)
    print("STEP 1: Getting answers from RAG API")
    print("=" * 60)

    results = []
    for idx, item in enumerate(test_items, 1):
        question = item.get("user_input", "")
        ground_truth = item.get("reference", "")

        print(f"[{idx}/{len(test_items)}] {question[:60]}...")

        response, contexts = call_rag_api(api_base_url, question, timeout)

        if response:
            print(f"  ✓ Got answer ({len(response)} chars, {len(contexts)} contexts)")
            results.append({
                "user_input": question,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            })
        else:
            print(f"  ✗ Failed - skipping")

    if not results:
        print("\nNo successful responses. Check if RAG API is running.")
        return 1

    # Step 2: Run RAGAS evaluation
    print("\n" + "=" * 60)
    print("STEP 2: Evaluating with RAGAS")
    print("=" * 60)

    # Initialize evaluator LLM and embeddings
    print(f"Initializing evaluator (LLM: {llm_model}, Embeddings: {embed_model})...")

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=api_key,
        temperature=0,
        model_kwargs={"response_mime_type": "application/json"}
    )

    langchain_embeddings = GoogleGenerativeAIEmbeddings(
        model=f"models/{embed_model}",
        google_api_key=api_key
    )
    embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    # Create dataset and metrics
    dataset = EvaluationDataset.from_list(results)
    metrics = [
        LLMContextPrecisionWithReference(llm=llm),
        LLMContextRecall(llm=llm),
        Faithfulness(llm=llm),
        AnswerCorrectness(llm=llm, embeddings=embeddings),
    ]

    print(f"Running {len(metrics)} metrics on {len(results)} samples...")
    print("(This may take a few minutes)\n")

    eval_results = evaluate(dataset=dataset, metrics=metrics)

    # Step 3: Show results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    df = eval_results.to_pandas()
    print_results(df)

    # Save results
    save_results(eval_results, df, args.out_json, args.out_csv)
    print(f"\nResults saved to {args.out_json} and {args.out_csv}")

    return 0


def call_rag_api(base_url: str, question: str, timeout: float) -> tuple:
    """Call RAG API and return (response, contexts)."""
    try:
        r = requests.post(
            f"{base_url}/v1/rag/query",
            json={"query": question},
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        response = data.get("response", "")
        contexts = [c.get("content", "") for c in data.get("citations", [])]
        return response, contexts
    except Exception as e:
        print(f"  Error: {e}")
        return None, []


def print_results(df: pd.DataFrame):
    """Print evaluation results."""
    metric_cols = [
        ('llm_context_precision_with_reference', 'Context Precision'),
        ('context_recall', 'Context Recall'),
        ('faithfulness', 'Faithfulness'),
        ('answer_correctness', 'Answer Correctness'),
    ]

    # Print per-sample results
    for i, row in df.iterrows():
        print(f"\nSample {i+1}: {row['user_input'][:50]}...")
        for col, label in metric_cols:
            if col in df.columns and pd.notna(row.get(col)):
                print(f"  {label:20s}: {row[col]:.2f}")
            else:
                print(f"  {label:20s}: N/A")

    # Print averages
    print("\n" + "-" * 60)
    print("AVERAGES")
    print("-" * 60)
    for col, label in metric_cols:
        if col in df.columns:
            avg = df[col].dropna().mean()
            if pd.notna(avg):
                print(f"  {label:20s}: {avg:.2f}")
            else:
                print(f"  {label:20s}: N/A")


def save_results(eval_results, df: pd.DataFrame, json_path: str, csv_path: str):
    """Save evaluation results to JSON and CSV files."""
    # Save JSON summary
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for col in ['llm_context_precision_with_reference', 'context_recall', 'faithfulness', 'answer_correctness']:
        if col in df.columns:
            avg = df[col].dropna().mean()
            summary[col] = avg if pd.notna(avg) else None

    with json_file.open("w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed CSV
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run RAG evaluation using RAGAS metrics")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config YAML file")
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET, help="Path to testset JSONL file")
    parser.add_argument("--out-json", type=str, default=DEFAULT_RESULTS_JSON, help="Output path for metrics JSON")
    parser.add_argument("--out-csv", type=str, default=DEFAULT_RESULTS_CSV, help="Output path for detailed CSV")
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_testset(path: str) -> List[Dict]:
    """Load test dataset from JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


if __name__ == "__main__":
    raise SystemExit(main())
