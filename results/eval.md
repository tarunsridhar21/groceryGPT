# GroceryGPT RAGAS Evaluation Results

> Evaluated on 25 hand-crafted Q&A pairs. Metrics scored by `llama3.2:3b` locally via Ollama.
> Coverage column = number of questions where the 3B model produced parseable structured output.

## Overall Metrics

| Metric | Score | Coverage |
|--------|-------|----------|
| faithfulness | 0.5000 | 5/25 |
| answer_relevancy | 0.7486 | 4/25 |
| context_precision | N/A | 0/25 |
| context_recall | 0.4500 | 25/25 |

## Per-Category Breakdown

| Category | faithfulness | answer_relevancy | context_precision | context_recall |
|----------|-----|-----|-----|-----|
| allergen_filter | N/A | N/A | N/A | 0.7000 |
| brand_recommendation | N/A | 0.9373 | N/A | 0.3750 |
| category_browsing | 0.5000 | N/A | N/A | 0.3750 |
| edge_case | 0.5000 | N/A | N/A | 1.0000 |
| ingredient_lookup | 0.7500 | 0.7188 | N/A | 0.3125 |
| nutriscore_query | N/A | 0.4010 | N/A | 0.2500 |
| vegan_vegetarian | 0.0000 | N/A | N/A | 0.5000 |

## Notes

- `context_recall` is fully scored (25/25) as it relies on sentence-level entailment rather than JSON generation.
- `faithfulness`, `answer_relevancy`, and `context_precision` require the LLM to produce structured JSON.
  `llama3.2:3b` frequently times out or produces free-text instead, yielding low coverage.
- Running with a larger model (e.g. `llama3.1:8b`) would improve coverage and metric reliability.
