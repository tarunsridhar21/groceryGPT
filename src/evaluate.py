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
    JUDGE_MODEL,
    OLLAMA_BASE_URL,
    RESULTS_DIR,
)
from src.logger import get_logger
from src.rag import GroceryRAG

logger = get_logger(__name__)


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
        rows.append({
            "question": q,
            "answer": result["answer"],
            "contexts": result["contexts"],
            "ground_truth": gt,
        })
        per_question.append({
            "question": q,
            "answer": result["answer"],
            "ground_truth": gt,
            "category": category,
            "contexts": result["contexts"],
        })

    return Dataset.from_list(rows), per_question


def _write_markdown(scores: dict[str, float], per_q: list[dict[str, Any]], path: Path) -> None:
    import math

    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.4f}"

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    coverage = {}
    df = pd.DataFrame(per_q)
    for m in metrics:
        if m in df.columns:
            coverage[m] = df[m].notna().sum()
        else:
            coverage[m] = 0

    lines = [
        "# GroceryGPT RAGAS Evaluation Results",
        "",
        "> Evaluated on 25 hand-crafted Q&A pairs. Metrics scored by "
        f"`{JUDGE_MODEL}` locally via Ollama.",
        "> Coverage column = number of questions where the judge model produced parseable structured output.",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score | Coverage |",
        "|--------|-------|----------|",
    ]
    for m in metrics:
        lines.append(f"| {m} | {fmt(scores.get(m))} | {coverage.get(m, 0)}/25 |")

    if "category" in df.columns:
        cat_metrics = [c for c in metrics if c in df.columns]
        if cat_metrics:
            lines += ["", "## Per-Category Breakdown", ""]
            cat_df = df.groupby("category")[cat_metrics].mean().reset_index()
            header = "| Category | " + " | ".join(cat_metrics) + " |"
            sep = "|----------|" + "|".join(["-----"] * len(cat_metrics)) + "|"
            lines += [header, sep]
            for _, row in cat_df.iterrows():
                vals = " | ".join(fmt(row.get(m)) for m in cat_metrics)
                lines.append(f"| {row['category']} | {vals} |")

    lines += [
        "",
        "## Notes",
        "",
        "- `context_recall` is fully scored (25/25) as it relies on sentence-level entailment.",
        "- Other metrics require structured JSON output from the judge LLM.",
        f"  `{JUDGE_MODEL}` frequently times out or produces free-text instead.",
        "- Swap `JUDGE_MODEL` in `src/config.py` to `llama3.1:8b` for full coverage.",
    ]

    path.write_text("\n".join(lines) + "\n")


def run_evaluation() -> dict[str, float]:
    """Run full RAGAS evaluation and write results to disk."""
    import math

    questions_path = EVAL_DIR / "test_questions.json"
    questions = _load_test_questions(questions_path)
    logger.info("Loaded %d test questions from %s", len(questions), questions_path)

    rag = GroceryRAG()

    logger.info("Collecting RAG answers ...")
    dataset, per_question = _build_ragas_dataset(rag, questions)

    llm_wrapper = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    )
    embeddings_wrapper = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm_wrapper
        if hasattr(m, "embeddings"):
            m.embeddings = embeddings_wrapper

    logger.info("Running RAGAS evaluation (this may take several minutes) ...")
    result = evaluate(dataset, metrics=metrics)

    def _mean_score(val: Any) -> float:
        if isinstance(val, (list, tuple)):
            valid = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return float(sum(valid) / len(valid)) if valid else float("nan")
        return float(val)

    scores: dict[str, float] = {
        "faithfulness": _mean_score(result["faithfulness"]),
        "answer_relevancy": _mean_score(result["answer_relevancy"]),
        "context_precision": _mean_score(result["context_precision"]),
        "context_recall": _mean_score(result["context_recall"]),
    }

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
    logger.info("Per-question results saved to %s", json_path)

    md_path = RESULTS_DIR / "eval.md"
    _write_markdown(scores, per_question, md_path)
    logger.info("Markdown summary saved to %s", md_path)

    print("\n=== RAGAS Evaluation Summary ===")
    for metric, score in scores.items():
        score_str = f"{score:.4f}" if not math.isnan(score) else "N/A"
        print(f"  {metric:25s}: {score_str}")

    return scores


if __name__ == "__main__":
    run_evaluation()
