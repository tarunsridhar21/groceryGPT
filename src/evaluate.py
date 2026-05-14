"""RAGAS evaluation of the RAG pipeline using local Ollama LLM and HuggingFace embeddings."""
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import (
    EMBEDDING_MODEL,
    EVAL_DIR,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    RESULTS_DIR,
)
from src.rag import GroceryRAG


def _load_test_questions(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _build_ragas_dataset(
    rag: GroceryRAG,
    questions: list[dict[str, Any]],
) -> tuple[Dataset, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_question: list[dict[str, Any]] = []

    for item in questions:
        q = item["question"]
        gt = item["ground_truth"]
        category = item.get("category", "general")

        result = rag.answer(q)
        rows.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": gt,
            }
        )
        per_question.append(
            {
                "question": q,
                "answer": result["answer"],
                "ground_truth": gt,
                "category": category,
                "contexts": result["contexts"],
            }
        )

    return Dataset.from_list(rows), per_question


def _write_markdown(scores: dict[str, float], per_q: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# GroceryGPT RAGAS Evaluation Results",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]
    for metric, score in scores.items():
        lines.append(f"| {metric} | {score:.4f} |")

    # Per-category breakdown
    df = pd.DataFrame(per_q)
    if "category" in df.columns and "faithfulness" in df.columns:
        lines += ["", "## Per-Category Breakdown", ""]
        cat_metrics = [c for c in df.columns if c in {
            "faithfulness", "answer_relevancy", "context_precision", "context_recall"
        }]
        if cat_metrics:
            cat_df = df.groupby("category")[cat_metrics].mean().reset_index()
            header = "| Category | " + " | ".join(cat_metrics) + " |"
            sep = "|----------|" + "|".join(["-----"] * len(cat_metrics)) + "|"
            lines += [header, sep]
            for _, row in cat_df.iterrows():
                vals = " | ".join(f"{row[m]:.4f}" for m in cat_metrics)
                lines.append(f"| {row['category']} | {vals} |")

    path.write_text("\n".join(lines) + "\n")


def run_evaluation() -> dict[str, float]:
    """Run full RAGAS evaluation and write results to disk."""
    questions_path = EVAL_DIR / "test_questions.json"
    questions = _load_test_questions(questions_path)
    print(f"Loaded {len(questions)} test questions from {questions_path}")

    rag = GroceryRAG()

    print("Collecting RAG answers ...")
    dataset, per_question = _build_ragas_dataset(rag, questions)

    llm_wrapper = LangchainLLMWrapper(
        ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm_wrapper
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings_wrapper

    print("Running RAGAS evaluation (this may take a few minutes) ...")
    result = evaluate(dataset, metrics=metrics)

    def _mean_score(val: Any) -> float:
        """Handle RAGAS returning either a scalar or a list of per-sample scores."""
        import math
        if isinstance(val, (list, tuple)):
            valid = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return float(sum(valid) / len(valid)) if valid else float("nan")
        v = float(val)
        return v

    scores: dict[str, float] = {
        "faithfulness": _mean_score(result["faithfulness"]),
        "answer_relevancy": _mean_score(result["answer_relevancy"]),
        "context_precision": _mean_score(result["context_precision"]),
        "context_recall": _mean_score(result["context_recall"]),
    }

    # Attach per-question scores if available
    try:
        result_df = result.to_pandas()
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            if col in result_df.columns:
                for i, val in enumerate(result_df[col].tolist()):
                    if i < len(per_question):
                        per_question[i][col] = float(val) if val is not None else None
    except Exception:
        pass

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "eval.json"
    with open(json_path, "w") as f:
        json.dump({"overall": scores, "per_question": per_question}, f, indent=2)
    print(f"Per-question results saved to {json_path}")

    md_path = RESULTS_DIR / "eval.md"
    _write_markdown(scores, per_question, md_path)
    print(f"Markdown summary saved to {md_path}")

    print("\n=== RAGAS Evaluation Summary ===")
    for metric, score in scores.items():
        print(f"  {metric:25s}: {score:.4f}")

    return scores


if __name__ == "__main__":
    run_evaluation()
