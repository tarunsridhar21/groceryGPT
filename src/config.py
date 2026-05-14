from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_DIR = PROJECT_ROOT / "eval"

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = "llama3.2:3b"
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

TOP_K: int = 5
PRODUCT_LIMIT: int = 2000
EMBED_BATCH_SIZE: int = 64
CHROMA_COLLECTION: str = "products"
