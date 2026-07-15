"""Offline retrieval evaluation (no LLM judge or API key required)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def reciprocal_rank(expected_source: str, retrieved: list[dict]) -> float:
    expected = expected_source.lower()
    for rank, item in enumerate(retrieved, start=1):
        if expected in str(item.get("source", "")).lower():
            return 1.0 / rank
    return 0.0


def recall_at_k(expected_source: str, retrieved: list[dict], k: int) -> float:
    expected = expected_source.lower()
    return float(any(expected in str(item.get("source", "")).lower() for item in retrieved[:k]))


def evaluate(rows: Iterable[dict], k: int = 5) -> dict:
    rows = list(rows)
    if not rows:
        return {"cases": 0, f"recall@{k}": 0.0, "mrr": 0.0}
    return {
        "cases": len(rows),
        f"recall@{k}": mean(recall_at_k(row["expected_source"], row["retrieved"], k) for row in rows),
        "mrr": mean(reciprocal_rank(row["expected_source"], row["retrieved"]) for row in rows),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported RAG retrieval results")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args()
    metrics = evaluate(load_jsonl(args.dataset), args.k)
    print(json.dumps(metrics, indent=2))
    if metrics[f"recall@{args.k}"] < args.min_recall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
