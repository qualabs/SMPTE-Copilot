import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
import pandas as pd

from ragas import evaluate, EvaluationDataset
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerCorrectness
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant as QdrantVectorStore


DEFAULT_TESTSET = "evaluations/synthetic/output/testset.jsonl"
DEFAULT_RESULTS_JSON = "evaluations/metrics/output/results.json"
DEFAULT_RESULTS_CSV = "evaluations/metrics/output/detailed.csv"
DEFAULT_K = 5


def load_yaml_config(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def resolve_google_api_key(cfg: dict) -> str | None:
    llm_cfg = (cfg.get("llm") or {}).get("llm_config") or {}
    if isinstance(llm_cfg, dict) and llm_cfg.get("api_key"):
        return llm_cfg.get("api_key")
    emb_cfg = (cfg.get("embedding") or {}).get("embed_config") or {}
    if isinstance(emb_cfg, dict) and emb_cfg.get("google_api_key"):
        return emb_cfg.get("google_api_key")
    return None


def resolve_models(cfg: dict) -> Tuple[str, str]:
    llm_model = ((cfg.get("llm") or {}).get("llm_config") or {}).get("model", "gemini-2.5-flash")
    embed_model = ((cfg.get("embedding") or {}).get("embed_config") or {}).get("model_name", "text-embedding-004")
    return llm_model, embed_model


def resolve_vectorstore_cfg(cfg: dict) -> dict:
    return cfg.get("vector_store") or {}


def resolve_retrieval_embed_model(cfg: dict) -> str:
    ret_cfg = cfg.get("retrieval") or {}
    return ret_cfg.get("retrieval_embed_model_name", "sentence-transformers/all-MiniLM-L6-v2")


def build_local_chroma(corpus_dir: str, embeddings) -> Chroma:
    texts: List[str] = []
    base = Path(corpus_dir)
    if base.exists() and base.is_dir():
        for ext in (".md", ".txt"):
            for fp in base.rglob(f"*{ext}"):
                try:
                    content = fp.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        texts.append(content)
                except Exception:
                    continue
    if not texts:
        texts = [""]
    return Chroma.from_texts(texts=texts, embedding=embeddings, collection_name="eval_temp")


def read_testset(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Testset not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def build_prompt(question: str, contexts: List[str]) -> str:
    header = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer cannot be determined from the context, say you don't know."
    )
    ctx = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts)])
    return f"{header}\n\n{ctx}\n\nQuestion: {question}\nAnswer:"


def init_generation_llm(api_key: str, llm_model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=llm_model, temperature=0, google_api_key=api_key)


def init_ragas_adapters(api_key: str, llm_model: str, embed_model: str) -> Tuple[Any, GoogleEmbeddings]:
    client = genai.Client(api_key=api_key)
    evaluator_llm = llm_factory(llm_model, provider="google", client=client)
    try:
        embeddings = GoogleEmbeddings(client=client, model=embed_model)
    except Exception:
        embeddings = GoogleEmbeddings(client=client, model="gemini-embedding-001")
    return evaluator_llm, embeddings


def build_retriever(cfg: dict, vs_cfg: dict, api_key: str, embed_model: str, k: int):
    store_name = (vs_cfg.get("store_name") or vs_cfg.get("name") or "").lower()
    if store_name == "qdrant":
        retrieval_embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)
    else:
        retrieval_model_name = resolve_retrieval_embed_model(cfg)
        retrieval_embeddings = HuggingFaceEmbeddings(model_name=retrieval_model_name)

    try:
        if store_name == "qdrant":
            url = (vs_cfg.get("store_config") or {}).get("url", "http://qdrant:6333")
            collection_name = (vs_cfg.get("store_config") or {}).get("collection_name", "rag_collection")
            client = QdrantClient(url=url, timeout=30.0, check_compatibility=False)
            vectordb = QdrantVectorStore(client=client, collection_name=collection_name, embeddings=retrieval_embeddings)
        else:
            persist_dir = (vs_cfg.get("persist_directory") or (vs_cfg.get("store_config") or {}).get("persist_directory") or "./vector_db")
            collection_name = (vs_cfg.get("collection_name") or (vs_cfg.get("store_config") or {}).get("collection_name") or "rag_collection")
            vectordb = Chroma(persist_directory=persist_dir, collection_name=collection_name, embedding_function=retrieval_embeddings)
    except Exception:
        corpus_dir = ((cfg.get("paths") or {}).get("markdown_dir") or "/app/data/markdown")
        vectordb = build_local_chroma(corpus_dir=corpus_dir, embeddings=retrieval_embeddings)

    return vectordb.as_retriever(search_kwargs={"k": k})


def generate_pipeline_outputs(llm: ChatGoogleGenerativeAI, retriever, test_items: List[Dict[str, Any]], cfg: dict, k: int):
    questions: List[str] = []
    answers: List[str] = []
    retrieved_contexts: List[List[str]] = []
    ground_truths: List[str] = []

    for s in test_items:
        q = s.get("user_input") or s.get("question") or ""
        gt = s.get("reference") or s.get("ground_truth") or ""
        questions.append(q)
        ground_truths.append(gt)

        try:
            docs = retriever.invoke(q)
        except Exception:
            corpus_dir = ((cfg.get("paths") or {}).get("markdown_dir") or "/app/data/markdown")
            vectordb = build_local_chroma(corpus_dir=corpus_dir, embeddings=retriever.embeddings)
            retr = vectordb.as_retriever(search_kwargs={"k": k})
            try:
                docs = retr.invoke(q)
            except Exception:
                docs = vectordb.similarity_search(q, k=k)

        ctxs = [d.page_content for d in docs]
        retrieved_contexts.append(ctxs)

        prompt = build_prompt(q, ctxs)
        resp = llm.invoke(prompt)
        ans_text = resp.content if hasattr(resp, "content") else (resp.text if hasattr(resp, "text") else str(resp))
        answers.append(ans_text)

    return questions, retrieved_contexts, answers, ground_truths


def build_eval_dataset(questions: List[str], contexts: List[List[str]], answers: List[str], gts: List[str]) -> Tuple[EvaluationDataset, List[Dict[str, Any]]]:
    samples = [
        {"user_input": q, "retrieved_contexts": c, "response": a, "reference": gt}
        for q, c, a, gt in zip(questions, contexts, answers, gts)
    ]
    return EvaluationDataset.from_list(samples), samples


def run_ragas_eval(dataset: EvaluationDataset, evaluator_llm, ragas_embeddings):
    return evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            AnswerCorrectness(llm=evaluator_llm, embeddings=ragas_embeddings),
        ],
        llm=evaluator_llm,
        embeddings=ragas_embeddings,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation on a testset using persisted vector store")
    parser.add_argument("--testset", type=str, default=DEFAULT_TESTSET, help="Path to testset JSONL")
    parser.add_argument("--out-json", type=str, default=DEFAULT_RESULTS_JSON, help="Where to write metrics JSON")
    parser.add_argument("--out-csv", type=str, default=DEFAULT_RESULTS_CSV, help="Where to write detailed CSV")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K contexts to retrieve")
    args = parser.parse_args()

    cfg = load_yaml_config("config.yaml")
    api_key = resolve_google_api_key(cfg)
    if not api_key:
        print("Google API key not found in config.yaml.")
        return 1

    llm_model, embed_model = resolve_models(cfg)
    vs_cfg = resolve_vectorstore_cfg(cfg)

    gen_llm = init_generation_llm(api_key, llm_model)
    evaluator_llm, ragas_embeddings = init_ragas_adapters(api_key, llm_model, embed_model)
    retriever = build_retriever(cfg, vs_cfg, api_key, embed_model, args.k)

    test_items = read_testset(args.testset)
    questions, contexts, answers, gts = generate_pipeline_outputs(gen_llm, retriever, test_items, cfg, args.k)
    eval_dataset, samples = build_eval_dataset(questions, contexts, answers, gts)
    results = run_ragas_eval(eval_dataset, evaluator_llm, ragas_embeddings)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        summary = results.scores if hasattr(results, "scores") else results
        json.dump(summary, f, indent=2)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = results.to_pandas()
    except Exception:
        try:
            df = pd.DataFrame(samples)
        except Exception:
            df = pd.DataFrame([])
    df.to_csv(out_csv, index=False)

    print(f"Saved metrics to {out_json} and details to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
