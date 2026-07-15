# RAGCare-QA retrieval baseline

Evaluation date: 2026-07-14

## Dataset audit

- Questions: 420
- Unique contexts: 406
- Unique normalized references: 169
- Missing question, answer, context, reference, or page: 0
- Missing `Type`: 270; specialty slices are therefore not reported

Each unique context is treated as one corpus document. The expected document is the context paired with the question. This evaluates retrieval only; it does not claim clinical answer correctness.

## Overall results

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median rank |
|---|---:|---:|---:|---:|---:|---:|
| Word TF-IDF (1–2 grams) | 62.62% | 81.90% | 84.52% | 88.81% | 0.7269 | 1 |
| Hybrid word + character TF-IDF | **68.81%** | **83.57%** | **86.90%** | **90.48%** | **0.7707** | 1 |

The hybrid score is `0.65 * word + 0.35 * character`. Character n-grams improve robustness to medical morphology, spelling variants, and bilingual Slovenian/English contexts.

## Hybrid slices

| Slice | Cases | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Basic complexity | 150 | 55.33% | 76.67% | 0.6536 |
| Intermediate complexity | 181 | 71.82% | 90.61% | 0.8023 |
| Advanced complexity | 89 | 85.39% | 96.63% | 0.9038 |
| Basic RAG | 315 | 68.25% | 86.03% | 0.7631 |
| Multi-vector RAG | 82 | 64.63% | 86.59% | 0.7500 |
| Graph-enhanced RAG | 23 | 91.30% | 100.00% | 0.9493 |

Advanced and graph-labelled questions score higher than basic questions. This should not be interpreted as graph retrieval performance: the evaluator uses the same lexical retriever for every row. The likely explanation is that advanced questions contain more discriminative medical terminology.


## Production hybrid retrieval

| Pipeline | Recall@1 | Recall@3 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|
| MiniLM + Chroma | 64.76% | 76.67% | 79.52% | 0.7092 |
| Lexical hybrid | 68.81% | 83.57% | 86.90% | 0.7636 |
| **Weighted RRF** | **69.29%** | 83.10% | **88.10%** | **0.7661** |
| Weighted RRF + BGE | 65.24% | 83.33% | 86.19% | 0.7382 |

The selected production configuration retrieves 20 candidates from MiniLM/Chroma and 20 from lexical TF-IDF, then fuses them with RRF weights `dense=0.35`, `lexical=1.0`, and rank constant `10`. BGE remains optional and defaults off because it reduced quality and added approximately 284 ms per question.
## Limitations and next experiment

1. This is an offline lexical baseline, not the application's MiniLM + Chroma + cross-encoder pipeline.
2. The corpus contains only the gold contexts. A production-like evaluation should add distractor passages from the complete source documents.
3. Several failures have ranks near the corpus size, suggesting almost no lexical overlap between the question and gold context.
4. Generation faithfulness, answer accuracy, citation correctness, safety, latency, and cost remain separate evaluations.

Run again with:

```bash
python -m evals.evaluate_ragcare
```

Machine-readable results and the 20 worst Recall@5 failures are in `evals/ragcare_results.json`.
