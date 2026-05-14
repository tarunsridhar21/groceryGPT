"""System health checks — Ollama reachability, ChromaDB index, raw data."""
from typing import Any

import requests

from src.config import CHROMA_COLLECTION, CHROMA_DIR, DATA_DIR, LLM_MODEL, OLLAMA_BASE_URL
from src.logger import get_logger

logger = get_logger(__name__)


def check_health() -> dict[str, Any]:
    """Return live status for all external dependencies."""
    status: dict[str, Any] = {
        "ollama_online": False,
        "ollama_model": LLM_MODEL,
        "index_built": False,
        "index_count": 0,
        "data_ingested": False,
    }

    status["data_ingested"] = (DATA_DIR / "products.parquet").exists()

    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(CHROMA_COLLECTION)
        count = collection.count()
        status["index_built"] = count > 0
        status["index_count"] = count
    except Exception as exc:
        logger.debug("ChromaDB health check failed: %s", exc)

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        status["ollama_online"] = resp.status_code == 200
    except Exception as exc:
        logger.debug("Ollama health check failed: %s", exc)

    return status
