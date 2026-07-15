"""Run the MMVA MiniLM -> Chroma -> BGE pipeline on RAGCare-QA."""

from __future__ import annotations

import argparse
import json
import ssl
import time
import uuid
from pathlib import Path
from statistics import mean

# Work around malformed entries in some Windows certificate stores. This is
# the same local-runtime workaround used by backend/main.py.
if hasattr(ssl.SSLContext, "_load_windows_store_certs"):
    ssl.SSLContext._load_windows_store_certs = lambda *args, **kwargs: None

import numpy as np
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.services.hybrid_retrieval import reciprocal_rank_fusion
from sentence_transformers import CrossEncoder


def metrics(rankings: list[list[int]], expected: list[int]) -> dict:
    ranks = []
    for result, target in zip(rankings, expected):
        ranks.append(result.index(target) + 1 if target in result else 0)
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    return {
        "cases": len(ranks),
        "recall@1": mean(rank == 1 for rank in ranks),
        "recall@3": mean(0 < rank <= 3 for rank in ranks),
        "recall@5": mean(0 < rank <= 5 for rank in ranks),
        "mrr@5": mean(reciprocal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("evals/RAGCare-QA.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/ragcare_dense_results.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=20)
    args = parser.parse_args()

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    contexts = list(dict.fromkeys(row["Context"].strip() for row in rows))
    context_to_id = {context: index for index, context in enumerate(contexts)}
    expected = [context_to_id[row["Context"].strip()] for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    started = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": args.batch_size, "normalize_embeddings": True},
    )
    vectorstore = Chroma.from_texts(
        texts=contexts,
        embedding=embeddings,
        metadatas=[{"context_id": index} for index in range(len(contexts))],
        collection_name=f"ragcare_eval_{uuid.uuid4().hex}",
    )
    index_seconds = time.perf_counter() - started

    lexical_started = time.perf_counter()
    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    word_docs = word.fit_transform(contexts)
    char_docs = char.fit_transform(contexts)
    lexical_index_seconds = time.perf_counter() - lexical_started

    retrieval_started = time.perf_counter()
    dense_rankings = []
    dense_docs = []
    lexical_docs = []
    fused_docs = []
    for row in rows:
        docs = vectorstore.similarity_search(row["Question"].strip(), k=args.candidate_k)
        dense_docs.append(docs)
        dense_rankings.append([int(doc.metadata["context_id"]) for doc in docs[:5]])
        question = row["Question"].strip()
        word_scores = (word.transform([question]) @ word_docs.T).toarray()[0]
        char_scores = (char.transform([question]) @ char_docs.T).toarray()[0]
        lexical_scores = 0.65 * word_scores + 0.35 * char_scores
        lexical_order = np.argsort(-lexical_scores, kind="stable")[:args.candidate_k]
        lexical = [Document(page_content=contexts[int(index)], metadata={"context_id": int(index)}) for index in lexical_order]
        lexical_docs.append(lexical)
        fused_docs.append(reciprocal_rank_fusion([docs, lexical], limit=args.candidate_k, rank_constant=10, weights=[0.35, 1.0]))
    retrieval_seconds = time.perf_counter() - retrieval_started

    reranker = CrossEncoder("BAAI/bge-reranker-base", device=device)
    rerank_started = time.perf_counter()
    reranked_fused = []
    for row, fused in zip(rows, fused_docs):
        fused_pairs = [(row["Question"].strip(), doc.page_content) for doc in fused]
        fused_scores = reranker.predict(fused_pairs, batch_size=args.batch_size, show_progress_bar=False)
        fused_order = np.argsort(-np.asarray(fused_scores), kind="stable")
        reranked_fused.append([int(fused[index].metadata["context_id"]) for index in fused_order[:5]])
    rerank_seconds = time.perf_counter() - rerank_started

    results = {
        "environment": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "questions": len(rows),
            "contexts": len(contexts),
            "candidate_k": args.candidate_k,
        },
        "timing_seconds": {
            "dense_index": index_seconds,
            "lexical_index": lexical_index_seconds,
            "retrieve_all": retrieval_seconds,
            "retrieve_per_question": retrieval_seconds / len(rows),
            "rerank_all": rerank_seconds,
            "rerank_per_question": rerank_seconds / len(rows),
        },
        "minilm_chroma": metrics(dense_rankings, expected),
        "lexical_hybrid": metrics([[int(doc.metadata["context_id"]) for doc in docs[:5]] for docs in lexical_docs], expected),
        "rrf": metrics([[int(doc.metadata["context_id"]) for doc in docs[:5]] for docs in fused_docs], expected),
        "rrf_bge": metrics(reranked_fused, expected),
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    vectorstore.delete_collection()


if __name__ == "__main__":
    main()
