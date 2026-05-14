"""LangChain RAG chain: retriever + Ollama LLM + prompt template."""
import argparse
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
    OLLAMA_BASE_URL,
    TOP_K,
)

_PROMPT_TEMPLATE = """\
You are a UK grocery assistant. Answer the user's question using ONLY the \
product information in the context below. If the answer is not in the context, \
say "I don't have that information in my product catalogue." Be concise. \
Cite product names you reference in square brackets.

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


class GroceryRAG:
    def __init__(self) -> None:
        device = _get_device()
        self._embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_collection(CHROMA_COLLECTION)

        self._llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,
        )

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict[str, Any]]:
        """Embed query and return top-k matching products."""
        embedding = self._embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        items: list[dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items

    def answer(self, query: str, k: int = TOP_K) -> dict[str, Any]:
        """Retrieve top-k products and generate an answer via the LLM."""
        sources = self.retrieve(query, k=k)
        contexts = [s["text"] for s in sources]
        context_str = "\n\n---\n\n".join(contexts)

        prompt = _PROMPT_TEMPLATE.format(context=context_str, question=query)
        response = self._llm.invoke(prompt)
        answer_text = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer_text, "sources": sources, "contexts": contexts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query GroceryGPT from the command line.")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of sources to retrieve")
    args = parser.parse_args()

    rag = GroceryRAG()
    result = rag.answer(args.query, k=args.k)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:")
    for i, src in enumerate(result["sources"], 1):
        name = src["metadata"].get("product_name", "Unknown")
        dist = src["distance"]
        print(f"  [{i}] {name} (distance: {dist:.4f})")
