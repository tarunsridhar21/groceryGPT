"""Builds a ChromaDB vector store from the ingested products parquet."""
import sys

import chromadb
import pandas as pd
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from src.logger import get_logger

logger = get_logger(__name__)


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_index() -> None:
    """Embed products and persist to ChromaDB."""
    parquet_path = DATA_DIR / "products.parquet"
    if not parquet_path.exists():
        logger.error("products.parquet not found at %s. Run `make ingest` first.", parquet_path)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d products from %s", len(df), parquet_path)

    device = _get_device()
    logger.info("Loading embedding model '%s' on device '%s'", EMBEDDING_MODEL, device)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if CHROMA_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(CHROMA_COLLECTION)
        logger.info("Dropped existing collection '%s'", CHROMA_COLLECTION)

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = df["text"].tolist()
    ids = df["code"].astype(str).tolist()
    metadatas = df[["product_name", "brands", "categories", "nutriscore_grade"]].to_dict(orient="records")

    total = len(texts)
    logger.info("Embedding %d products in batches of %d", total, EMBED_BATCH_SIZE)

    for start in tqdm(range(0, total, EMBED_BATCH_SIZE), desc="Indexing"):
        end = min(start + EMBED_BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_meta,
        )

    logger.info("Index built. Collection '%s' contains %d items.", CHROMA_COLLECTION, collection.count())


if __name__ == "__main__":
    build_index()
