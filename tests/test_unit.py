"""Unit tests — CI-safe, no external services required."""
import pandas as pd
import pytest

from src.config import (
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    JUDGE_MODEL,
    LLM_MODEL,
    PRODUCT_LIMIT,
    TOP_K,
)
from src.ingest import _build_text


def test_config_constants() -> None:
    assert EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert LLM_MODEL == "llama3.2:3b"
    assert JUDGE_MODEL == "llama3.2:3b"
    assert TOP_K == 5
    assert PRODUCT_LIMIT == 2000
    assert CHROMA_COLLECTION == "products"


def test_build_text_all_fields() -> None:
    row = pd.Series({
        "product_name": "Test Biscuit",
        "brands": "TestBrand",
        "categories": "Biscuits",
        "ingredients_text": "Flour, Sugar, Butter",
        "allergens": "en:gluten,en:milk",
        "labels": "Vegetarian",
        "nutriscore_grade": "c",
        "nova_group": "4",
        "quantity": "200g",
    })
    text = _build_text(row)
    assert "Test Biscuit" in text
    assert "TestBrand" in text
    assert "Flour, Sugar, Butter" in text
    assert "en:gluten" in text
    assert "Nutri-Score: c" in text
    assert "NOVA group: 4" in text
    assert "Quantity: 200g" in text


def test_build_text_missing_fields_do_not_raise() -> None:
    row = pd.Series({"product_name": "Minimal"})
    text = _build_text(row)
    assert "Minimal" in text
    assert "Product:" in text


def test_build_text_structure() -> None:
    row = pd.Series({
        "product_name": "A",
        "brands": "B",
        "categories": "C",
        "ingredients_text": "D",
        "allergens": "E",
        "labels": "F",
        "nutriscore_grade": "a",
        "nova_group": "1",
        "quantity": "100g",
    })
    lines = _build_text(row).splitlines()
    assert lines[0].startswith("Product:")
    assert lines[1].startswith("Brand:")
    assert lines[4].startswith("Allergens:")
    assert lines[6].startswith("Nutri-Score:")
