"""Pure helpers for deterministic hybrid retrieval."""

from __future__ import annotations

from typing import Any, Iterable, Sequence


def document_key(document: Any) -> str:
    metadata = getattr(document, "metadata", {}) or {}
    if "context_id" in metadata:
        return f"context:{metadata['context_id']}"
    source = str(metadata.get("source", ""))
    chunk = str(metadata.get("chunk", ""))
    if source or chunk:
        return f"source:{source}|chunk:{chunk}"
    return f"content:{getattr(document, 'page_content', str(document))}"


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]],
    *,
    limit: int,
    rank_constant: int = 60,
    weights: Iterable[float] | None = None,
) -> list[Any]:
    """Fuse ranked lists without requiring comparable retrieval scores."""
    ranking_weights = list(weights or [1.0] * len(rankings))
    if len(ranking_weights) != len(rankings):
        raise ValueError("weights must match rankings")

    scores: dict[str, float] = {}
    documents: dict[str, Any] = {}
    first_seen: dict[str, int] = {}
    seen_order = 0
    for weight, ranking in zip(ranking_weights, rankings):
        for rank, document in enumerate(ranking, start=1):
            key = document_key(document)
            documents.setdefault(key, document)
            if key not in first_seen:
                first_seen[key] = seen_order
                seen_order += 1
            scores[key] = scores.get(key, 0.0) + weight / (rank_constant + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    result = []
    for key in ordered[:limit]:
        document = documents[key]
        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata["rrf_score"] = scores[key]
        try:
            document.metadata = metadata
        except Exception:
            pass
        result.append(document)
    return result
