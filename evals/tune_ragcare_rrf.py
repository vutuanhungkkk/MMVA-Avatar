"""Tune RRF weights without reranking or changing retrieval models."""

from __future__ import annotations

import json
import ssl
from pathlib import Path
from statistics import mean

if hasattr(ssl.SSLContext, "_load_windows_store_certs"):
    ssl.SSLContext._load_windows_store_certs = lambda *args, **kwargs: None

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def score(rankings: list[list[int]], expected: list[int]) -> dict:
    ranks = [ranking.index(target) + 1 if target in ranking else 0 for ranking, target in zip(rankings, expected)]
    return {
        "recall@1": mean(rank == 1 for rank in ranks),
        "recall@5": mean(0 < rank <= 5 for rank in ranks),
        "mrr@5": mean(1 / rank if rank else 0 for rank in ranks),
    }


def fuse(dense: np.ndarray, lexical: np.ndarray, dense_weight: float, rank_constant: int, limit: int = 5) -> list[int]:
    scores: dict[int, float] = {}
    for weight, ranking in ((dense_weight, dense), (1.0, lexical)):
        for rank, document_id in enumerate(ranking, start=1):
            document_id = int(document_id)
            scores[document_id] = scores.get(document_id, 0.0) + weight / (rank_constant + rank)
    return sorted(scores, key=lambda item: -scores[item])[:limit]


def main() -> None:
    rows = json.loads(Path("evals/RAGCare-QA.json").read_text(encoding="utf-8"))
    contexts = list(dict.fromkeys(row["Context"].strip() for row in rows))
    context_to_id = {context: index for index, context in enumerate(contexts)}
    questions = [row["Question"].strip() for row in rows]
    expected = [context_to_id[row["Context"].strip()] for row in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    context_embeddings = model.encode(contexts, batch_size=32, normalize_embeddings=True)
    question_embeddings = model.encode(questions, batch_size=32, normalize_embeddings=True)
    dense_rankings = np.argsort(-(question_embeddings @ context_embeddings.T), axis=1)[:, :20]

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    word_docs, char_docs = word.fit_transform(contexts), char.fit_transform(contexts)
    lexical_scores = 0.65 * (word.transform(questions) @ word_docs.T).toarray()
    lexical_scores += 0.35 * (char.transform(questions) @ char_docs.T).toarray()
    lexical_rankings = np.argsort(-lexical_scores, axis=1)[:, :20]

    results = []
    for rank_constant in (10, 30, 60):
        for dense_weight in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            rankings = [
                fuse(dense, lexical, dense_weight, rank_constant)
                for dense, lexical in zip(dense_rankings, lexical_rankings)
            ]
            results.append({
                "dense_weight": dense_weight,
                "lexical_weight": 1.0,
                "rank_constant": rank_constant,
                **score(rankings, expected),
            })
    results.sort(key=lambda item: (item["recall@5"], item["mrr@5"], item["recall@1"]), reverse=True)
    Path("evals/ragcare_rrf_tuning.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results[:10], indent=2))


if __name__ == "__main__":
    main()
