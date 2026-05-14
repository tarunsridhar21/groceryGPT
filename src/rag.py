"""LangChain RAG chain: retriever + Ollama LLM + prompt template."""
import argparse
import datetime
import json
import time
from typing import Any

import chromadb
import torch
from chromadb.config import Settings
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LOGS_DIR,
    OLLAMA_BASE_URL,
    TOP_K,
)
from src.logger import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """\
You are a helpful UK grocery assistant. Answer the user's question using \
the product information provided in the context below.

Guidelines:
- You MAY infer dietary suitability from ingredients and allergen fields. \
For example: a product with no meat, fish, dairy, eggs or honey in its \
ingredients is likely suitable for vegans. A product with no wheat, barley, \
rye or oats is likely gluten-free.
- If a product's Labels field contains terms like "vegan", "vegetarian", \
"organic", or "gluten-free", treat those as confirmed facts.
- Only say "I don't have that information in my product catalogue." if the \
context contains no products relevant to the question at all.
- Be concise. Cite product names in square brackets.

Context:
{context}

Question: {question}

Answer:"""


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _log_query(
    query: str,
    answer: str,
    latency_ms: float,
    n_sources: int,
    where: dict | None,
) -> None:
    """Append one line to logs/queries.jsonl for audit and analysis."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "query": query,
            "answer": answer,
            "latency_ms": round(latency_ms, 1),
            "n_sources": n_sources,
            "where_filter": where,
        }
        with open(LOGS_DIR / "queries.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write query log: %s", exc)


class GroceryRAG:
    def __init__(self) -> None:
        device = _get_device()
        logger.info("Loading embedding model '%s' on device '%s'", EMBEDDING_MODEL, device)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_collection(CHROMA_COLLECTION)
        logger.info("Connected to ChromaDB collection '%s' (%d items)", CHROMA_COLLECTION, self._collection.count())

        self._llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )
        logger.info("LLM ready: %s @ %s", LLM_MODEL, OLLAMA_BASE_URL)

    def retrieve(self, query: str, k: int = TOP_K, where: dict | None = None) -> list[dict[str, Any]]:
        """Embed query and return top-k matching products."""
        embedding = self._embedder.encode(query, normalize_embeddings=True).tolist()

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        items: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items

    # Cosine distance threshold: ChromaDB reports cosine distance in [0, 2].
    # Distance > 0.7 means cosine similarity < 0.3 — very poor match.
    # If *all* top-k results are below this relevance bar, skip the LLM call
    # entirely and return the standard decline response immediately.
    _OFF_CATALOGUE_THRESHOLD = 0.7

    def answer(self, query: str, k: int = TOP_K, where: dict | None = None) -> dict[str, Any]:
        """Retrieve top-k products and generate an answer via the LLM.

        Short-circuits before the LLM if every retrieved result has cosine
        distance > _OFF_CATALOGUE_THRESHOLD — i.e. nothing in the index is
        meaningfully related to the query.
        """
        t0 = time.perf_counter()

        sources = self.retrieve(query, k=k, where=where)
        retrieve_ms = (time.perf_counter() - t0) * 1000

        # ── Off-catalogue guard ────────────────────────────────────────────────
        if sources and all(s["distance"] > self._OFF_CATALOGUE_THRESHOLD for s in sources):
            total_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "query=%r off-catalogue (min_dist=%.3f) retrieve_ms=%.0f — skipping LLM",
                query[:80], min(s["distance"] for s in sources), retrieve_ms,
            )
            answer_text = "I don't have that information in my product catalogue."
            _log_query(query, answer_text, total_ms, len(sources), where)
            return {
                "answer": answer_text,
                "sources": sources,
                "contexts": [],
                "latency_ms": total_ms,
            }

        contexts = [s["text"] for s in sources]
        context_str = "\n\n---\n\n".join(contexts)

        prompt = _PROMPT_TEMPLATE.format(context=context_str, question=query)

        t1 = time.perf_counter()
        response = self._llm.invoke(prompt)
        llm_ms = (time.perf_counter() - t1) * 1000

        answer_text = response.content if hasattr(response, "content") else str(response)
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "query=%r sources=%d retrieve_ms=%.0f llm_ms=%.0f total_ms=%.0f",
            query[:80], len(sources), retrieve_ms, llm_ms, total_ms,
        )

        _log_query(query, answer_text, total_ms, len(sources), where)

        return {
            "answer": answer_text,
            "sources": sources,
            "contexts": contexts,
            "latency_ms": total_ms,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query GroceryGPT from the command line.")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of sources to retrieve")
    args = parser.parse_args()

    rag = GroceryRAG()
    result = rag.answer(args.query, k=args.k)
    print("\nAnswer:\n", result["answer"])
    print(f"\nLatency: {result['latency_ms']:.0f} ms")
    print("\nSources:")
    for i, src in enumerate(result["sources"], 1):
        name = src["metadata"].get("product_name", "Unknown")
        dist = src["distance"]
        print(f"  [{i}] {name} (distance: {dist:.4f})")
