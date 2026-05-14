# GroceryGPT — RAG-based UK Grocery Intelligence Assistant

RAG over ~1,900 UK grocery products from Open Food Facts. Local LLM, embedded vector store, RAGAS-evaluated.

## Demo

https://github.com/user-attachments/assets/demo_full.mp4

> **2m 20s walkthrough** — 10 scenes: system health, allergen filtering, Nutri-Score constraints,
> vegan inference, brand deep-dive, cross-category ingredient lookup, edge-case handling,
> latency logging, and the RAGAS evaluation architecture.
>
> *Can't see the video above? [Download MP4 (3.8 MB)](docs/demo_full.mp4) or [download GIF (19 MB)](docs/demo_full.gif).*

## Architecture

```mermaid
flowchart LR
    A[User] --> B[Streamlit UI]
    B --> C[Sentence-Transformers Embedder]
    C --> D[ChromaDB Vector Store]
    D --> E[Top-K Retrieved Products]
    E --> F[LangChain RAG Chain]
    F --> G[Ollama llama3.2:3b]
    G --> H[Answer + Sources]
    H --> B
```

## Stack

| Component | Library / Version | Role |
|-----------|-------------------|------|
| LLM | Ollama `llama3.2:3b` | Answer generation (fully local) |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Dense retrieval embeddings |
| Vector Store | ChromaDB (cosine distance) | ANN product index |
| RAG Orchestration | LangChain + langchain-ollama | Chain management |
| Evaluation | RAGAS | Faithfulness, relevancy, precision, recall |
| UI | Streamlit | Interactive chat interface |
| Data | Open Food Facts API v2 | ~1,900 UK products |
| Acceleration | Apple MPS / CUDA / CPU | Device-adaptive embedding |

## Evaluation Results

All evaluation runs locally using RAGAS 0.4.x with `llama3.2:3b` as the judge LLM and
`BAAI/bge-small-en-v1.5` for semantic similarity. Test set: 25 hand-crafted Q&A pairs
across 7 categories (see `eval/test_questions.json`).

| Metric | Score | Coverage | Mechanism |
|--------|-------|----------|-----------|
| context_recall | 0.4500 | 25/25 | Sentence-level NLI entailment — does not require structured JSON from the judge |
| faithfulness | 0.5000 | 5/25 | LLM-as-judge (structured JSON output) |
| answer_relevancy | 0.7486 | 4/25 | LLM-as-judge (structured JSON output) |
| context_precision | N/A | 0/25 | LLM-as-judge (structured JSON output) |

**Coverage note:** Faithfulness, answer_relevancy, and context_precision require the judge
LLM to return valid JSON objects. `llama3.2:3b` (3B parameters, 4-bit quantised) fails this
reliably at ~10s timeouts per sub-prompt. `context_recall` uses sentence-level NLI instead
and achieves full 25/25 coverage. Re-running `make eval` with `llama3.1:8b` as the judge
model would restore full coverage for all metrics — see `src/evaluate.py` for the
`JUDGE_MODEL` config variable.

### Per-category context_recall breakdown

| Category | Score | n |
|----------|-------|---|
| edge_case | 1.0000 | 1 |
| allergen_filter | 0.7000 | 5 |
| vegan_vegetarian | 0.5000 | 3 |
| category_browsing | 0.3750 | 4 |
| brand_recommendation | 0.3750 | 4 |
| ingredient_lookup | 0.3125 | 5 |
| nutriscore_query | 0.2500 | 4 |
| **Overall** | **0.4500** | **25** |

Nutri-Score queries score lowest because `llama3.2:3b` hedges on specific grade values
rather than citing the retrieved data directly.

## Quickstart

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running,
`llama3.2:3b` pulled (`ollama pull llama3.2:3b`).

```bash
git clone https://github.com/tarunsridhar21/groceryGPT.git
cd grocerygpt
make setup        # creates .venv and installs dependencies
make ingest       # fetches ~1,900 UK products from Open Food Facts (~5 min)
make index        # embeds products into ChromaDB (~30 sec on Apple MPS)
make app          # launches Streamlit chat UI at http://localhost:8501
```

To run the full evaluation (requires Ollama running, ~16 min):
```bash
make eval
```

To run all steps in sequence:
```bash
make all
```

## Evaluation Methodology

### Test Set Construction

The `eval/test_questions.json` file contains 25 hand-crafted question-answer pairs covering seven categories:

- **ingredient_lookup** — asks what a product contains
- **allergen_filter** — queries for products safe for specific dietary restrictions
- **brand_recommendation** — asks about specific well-known UK brands
- **nutriscore_query** — asks about nutritional quality scores
- **vegan_vegetarian** — asks about plant-based or animal-free products
- **category_browsing** — browses product types
- **edge_case** — queries for products that do not exist in the catalogue

Ground truths are phrased generically to match realistic Open Food Facts data and avoid hardcoding specific product codes.

### RAGAS Metrics

All evaluation runs entirely locally using `llama3.2:3b` via Ollama and `BAAI/bge-small-en-v1.5` for embeddings — zero paid API calls.

| Metric | What it measures |
|--------|-----------------|
| **Faithfulness** | Whether the generated answer is grounded in the retrieved context (no hallucinations). Score: 0–1, higher is better. |
| **Answer Relevancy** | How well the answer addresses the question. Penalises verbose or off-topic answers. Score: 0–1. |
| **Context Precision** | What fraction of the retrieved contexts are actually relevant to answering the question. Measures retrieval precision. |
| **Context Recall** | Whether the retrieved contexts contain all the information needed to answer the question. Measures retrieval coverage. |

A high faithfulness + low context recall combination indicates the retriever is the bottleneck. A low faithfulness with adequate context suggests the LLM is drifting from the source material.

## Limitations and Next Steps

| Limitation | Possible improvement |
|------------|---------------------|
| ~1,900-product sample (full OFF UK dataset is ~800k) | Nightly sync with full OFF dump; distributed indexing |
| No query rewriting or HyDE | Add hypothetical document embeddings or LLM-based query expansion |
| No cross-encoder re-ranking | Add a `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker after retrieval |
| Single-turn conversation only | Add LangChain `ConversationBufferMemory` for multi-turn context |
| No fine-tuning | Fine-tune embedding model on grocery-specific query-product pairs |
| Ollama latency on CPU (~5s/query) | Quantised GGUF on GPU or switch to a smaller model for demo |

## Licence

MIT
