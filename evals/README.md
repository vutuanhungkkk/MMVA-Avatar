# RAG evaluation

The JSONL format keeps the golden question, expected source, and ranked retrieval output. It deliberately evaluates retrieval without an LLM judge, API key, or GPU.

```bash
python -m evals.run_rag_eval evals/sample_retrieval_results.jsonl --k 5 --min-recall 0.8
```

Replace the sample with reviewed medical questions. Export both baseline and reranked results, then compare Recall@K and MRR. Generation quality should be reviewed separately for faithfulness, citation correctness, safety, latency, and cost.
