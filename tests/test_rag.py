"""Sanity tests for the GroceryGPT RAG pipeline."""
import pytest

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_COLLECTION, CHROMA_DIR
from src.rag import GroceryRAG


@pytest.fixture(scope="module")
def rag() -> GroceryRAG:
    return GroceryRAG()


def test_collection_not_empty() -> None:
    """ChromaDB collection must have been populated by build_index."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(CHROMA_COLLECTION)
    assert collection.count() > 0, "ChromaDB collection is empty — run `make index` first"


def test_retrieve_returns_k_results(rag: GroceryRAG) -> None:
    """retrieve() should return exactly k documents."""
    results = rag.retrieve("biscuits", k=3)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"


def test_answer_returns_non_empty_string(rag: GroceryRAG) -> None:
    """answer() should produce a non-empty string answer."""
    result = rag.answer("What allergens are in a popular biscuit brand?")
    assert isinstance(result["answer"], str), "Answer is not a string"
    assert len(result["answer"].strip()) > 0, "Answer is empty"
