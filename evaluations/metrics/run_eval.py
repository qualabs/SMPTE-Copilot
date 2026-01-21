"""
RAG Evaluation Script
Evaluates a RAG pipeline using RAGAS metrics against a test dataset.
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

import yaml
import pandas as pd
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant as QdrantVectorStore

from ragas import evaluate, EvaluationDataset
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerCorrectness
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings


# Configuration defaults
DEFAULT_CONFIG = "evaluations/config.yaml"
DEFAULT_TESTSET = "evaluations/synthetic/output/testset.jsonl"
DEFAULT_RESULTS_JSON = "evaluations/metrics/output/results.json"
DEFAULT_RESULTS_CSV = "evaluations/metrics/output/detailed.csv"


def main() -> int:
    """Main entry point for RAG evaluation."""
    args = parse_arguments()
    config = load_config(args.config)
    
    # Initialize components
    api_key = config["llm"]["llm_config"]["api_key"]
    llm_model = config["llm"]["llm_config"]["model"]
    embed_model = config["embedding"]["embed_config"]["model_name"]
    
    generation_llm = create_generation_llm(api_key, llm_model)
    evaluator_llm = create_evaluator_llm(api_key, llm_model)
    ragas_embeddings = create_ragas_embeddings(api_key, embed_model)
    retriever = create_retriever(config, api_key, embed_model, args.k)
    
    # Run evaluation pipeline
    test_items = load_testset(args.testset)
    questions, contexts, answers, ground_truths = generate_answers(
        generation_llm, retriever, test_items, args.k
    )
    
    eval_dataset = create_eval_dataset(questions, contexts, answers, ground_truths)
    results = evaluate_with_ragas(eval_dataset, evaluator_llm, ragas_embeddings)
    
    # Save results
    save_results(results, eval_dataset, args.out_json, args.out_csv)
    print(f"Evaluation complete. Results saved to {args.out_json} and {args.out_csv}")
    
    return 0


def generate_answers(llm, retriever, test_items: List[Dict], k: int):
    """Generate answers for test questions using the RAG pipeline."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in test_items:
        question = item.get("user_input", "")
        ground_truth = item.get("reference", "")
        
        questions.append(question)
        ground_truths.append(ground_truth)
        
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        retrieved_contexts = [doc.page_content for doc in docs]
        contexts.append(retrieved_contexts)
        
        # Generate answer
        prompt = create_prompt(question, retrieved_contexts)
        response = llm.invoke(prompt)
        answer = extract_answer_text(response)
        answers.append(answer)
    
    return questions, contexts, answers, ground_truths


def evaluate_with_ragas(dataset: EvaluationDataset, evaluator_llm, embeddings):
    """Run RAGAS evaluation on the dataset."""
    return evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            AnswerCorrectness(llm=evaluator_llm, embeddings=embeddings),
        ],
        llm=evaluator_llm,
        embeddings=embeddings,
    )


def save_results(results, dataset: EvaluationDataset, json_path: str, csv_path: str):
    """Save evaluation results to JSON and CSV files."""
    # Save JSON metrics
    json_file = Path(json_path)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w") as f:
        metrics = results.scores if hasattr(results, "scores") else results
        json.dump(metrics, f, indent=2)
    
    # Save detailed CSV
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df = results.to_pandas() if hasattr(results, "to_pandas") else pd.DataFrame()
    df.to_csv(csv_file, index=False)


# Component initialization functions

def create_generation_llm(api_key: str, model: str):
    """Initialize the LLM for answer generation."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=api_key
    )


def create_evaluator_llm(api_key: str, model: str):
    """Initialize the LLM for RAGAS evaluation."""
    client = genai.Client(api_key=api_key)
    return llm_factory(model, provider="google", client=client)


def create_ragas_embeddings(api_key: str, model: str):
    """Initialize embeddings for RAGAS evaluation."""
    client = genai.Client(api_key=api_key)
    return GoogleEmbeddings(client=client, model=model)


def create_retriever(config: dict, api_key: str, embed_model: str, k: int):
    """Initialize the vector store retriever."""
    vs_config = config["vector_store"]
    embeddings = GoogleGenerativeAIEmbeddings(
        model=embed_model,
        google_api_key=api_key
    )
    
    if vs_config["store_name"].lower() == "qdrant":
        client = QdrantClient(
            url=vs_config["store_config"]["url"],
            timeout=30.0,
            check_compatibility=False
        )
        vectordb = QdrantVectorStore(
            client=client,
            collection_name=vs_config["store_config"]["collection_name"],
            embeddings=embeddings
        )
    else:
        raise ValueError(f"Unsupported vector store: {vs_config['store_name']}")
    
    return vectordb.as_retriever(search_kwargs={"k": k})


def create_eval_dataset(questions: List[str], contexts: List[List[str]], 
                       answers: List[str], ground_truths: List[str]) -> EvaluationDataset:
    """Create RAGAS evaluation dataset from pipeline outputs."""
    samples = [
        {
            "user_input": q,
            "retrieved_contexts": c,
            "response": a,
            "reference": gt
        }
        for q, c, a, gt in zip(questions, contexts, answers, ground_truths)
    ]
    return EvaluationDataset.from_list(samples)


# Helper functions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation using RAGAS metrics"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--testset",
        type=str,
        default=DEFAULT_TESTSET,
        help="Path to testset JSONL file"
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=DEFAULT_RESULTS_JSON,
        help="Output path for metrics JSON"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=DEFAULT_RESULTS_CSV,
        help="Output path for detailed CSV"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of contexts to retrieve"
    )
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


def create_prompt(question: str, contexts: List[str]) -> str:
    """Create a prompt for answer generation."""
    header = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer cannot be determined from the context, say you don't know."
    )
    context_text = "\n\n".join(
        f"[Context {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    return f"{header}\n\n{context_text}\n\nQuestion: {question}\nAnswer:"


def extract_answer_text(response) -> str:
    """Extract text from LLM response."""
    if hasattr(response, "content"):
        return response.content
    if hasattr(response, "text"):
        return response.text
    return str(response)


if __name__ == "__main__":
    raise SystemExit(main())
