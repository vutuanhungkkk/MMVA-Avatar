# Engineering evidence

## Session isolation

The browser persists a random `session_id`. HTTP sends it through `X-Session-ID`; WebSocket sends it as a query parameter. `SessionManager` owns a separate `Orchestrator` and conversation history for every ID and expires idle sessions after one hour.

The vector database remains a shared knowledge resource. Conversation memory and provider instances do not.

## RAG evidence and evaluation

Retrieved passages are labelled `[source, page/chunk]`. The system prompt treats document text as untrusted evidence and tells the model to preserve these labels in factual claims.

Run the deterministic retrieval gate without an API key:

```bash
python -m evals.run_rag_eval evals/sample_retrieval_results.jsonl --min-recall 0.8
```

The evaluator reports Recall@K and MRR. Add reviewed cases rather than tuning against the sample dataset.

## Security and observability

- Uploads have server-generated storage names, an extension allowlist, and a 20 MB streaming limit.
- Provider names are schema constrained.
- CORS is configured with `CORS_ORIGINS` and defaults to the local frontend only.
- Model and user text are appended as text nodes, preventing HTML execution in chat messages.
- Every HTTP response returns `X-Request-ID`; structured request logs include status and duration.
- `/health/live` and `/health/ready` support deployment probes.

## Quality gates

GitHub Actions runs critical-error linting, tests with coverage, the deterministic RAG threshold, Bandit, and dependency auditing. Development dependencies are intentionally separate from the GPU-heavy runtime environment.


## Hybrid retrieval decision

MMVA fuses MiniLM/Chroma top-20 and word+character TF-IDF top-20 using weighted RRF (dense=0.35, lexical=1.0, k=10). On RAGCare-QA this reached Recall@5 88.10%, versus 86.90% lexical and 79.52% dense. BGE is retained behind RAG_ENABLE_RERANKER=true but defaults off because it reduced Recall@5 to 86.19% and added approximately 284 ms/question.
