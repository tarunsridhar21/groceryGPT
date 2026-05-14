"""Integration tests — require Ollama running and ChromaDB index built.

Run with:
    make test                        # all tests
    pytest -m "not integration"      # skip these (CI-safe)
"""
import pytest

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION, CHROMA_DIR
from src.rag import GroceryRAG


@pytest.fixture(scope="module")
def rag() -> GroceryRAG:
    return GroceryRAG()


@pytest.mark.integration
def test_collection_not_empty() -> None:
    """ChromaDB collection must have been populated by build_index."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(CHROMA_COLLECTION)
    assert collection.count() > 0, "ChromaDB collection is empty — run `make index` first"


@pytest.mark.integration
def test_retrieve_returns_k_results(rag: GroceryRAG) -> None:
    """retrieve() should return exactly k documents."""
    results = rag.retrieve("biscuits", k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"


@pytest.mark.integration
def test_answer_returns_non_empty_string(rag: GroceryRAG) -> None:
    """answer() should produce a non-empty string answer and include latency."""
    result = rag.answer("What allergens are in a popular biscuit brand?")
    assert isinstance(result["answer"], str), "Answer is not a string"
    assert len(result["answer"].strip()) > 0, "Answer is empty"
    assert "latency_ms" in result, "latency_ms missing from result"
    assert result["latency_ms"] > 0, "latency_ms should be positive"
