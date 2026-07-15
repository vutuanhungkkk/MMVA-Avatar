"""Evaluate deterministic retrieval baselines on the RAGCare-QA benchmark."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _metrics(ranks: list[int]) -> dict[str, float | int]:
    return {
        "cases": len(ranks),
        "recall@1": mean(rank <= 1 for rank in ranks),
        "recall@3": mean(rank <= 3 for rank in ranks),
        "recall@5": mean(rank <= 5 for rank in ranks),
        "recall@10": mean(rank <= 10 for rank in ranks),
        "mrr": mean(1.0 / rank for rank in ranks),
        "median_rank": int(np.median(ranks)),
    }


def evaluate(rows: list[dict]) -> dict:
    # Identical contexts are one corpus document and one valid retrieval target.
    contexts = list(dict.fromkeys(row["Context"].strip() for row in rows))
    context_to_id = {context: index for index, context in enumerate(contexts)}
    questions = [row["Question"].strip() for row in rows]

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
    word_docs = word.fit_transform(contexts)
    word_queries = word.transform(questions)

    # Character n-grams make the baseline less brittle to medical morphology,
    # spelling variants, and the Slovenian/English contexts in RAGCare-QA.
    char = TfidfVectorizer(analyzer="char_wb", lowercase=True, ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    char_docs = char.fit_transform(contexts)
    char_queries = char.transform(questions)

    word_scores = (word_queries @ word_docs.T).toarray()
    char_scores = (char_queries @ char_docs.T).toarray()
    hybrid_scores = 0.65 * word_scores + 0.35 * char_scores

    outputs = {}
    for name, scores in {"word_tfidf": word_scores, "hybrid_tfidf": hybrid_scores}.items():
        ranks = []
        groups: dict[str, list[int]] = defaultdict(list)
        failures = []
        for index, row in enumerate(rows):
            expected = context_to_id[row["Context"].strip()]
            order = np.argsort(-scores[index], kind="stable")
            rank = int(np.flatnonzero(order == expected)[0]) + 1
            ranks.append(rank)
            groups[f"complexity:{row['Complexity'].strip().lower()}"] .append(rank)
            groups[f"pipeline:{row['RAG Pipeline'].strip()}"] .append(rank)
            if rank > 5:
                failures.append({
                    "row": index,
                    "rank": rank,
                    "question": row["Question"].strip(),
                    "expected_reference": row["Reference"].strip(),
                    "expected_page": row["Page"].strip(),
                    "top_reference": rows[next(
                        i for i, candidate in enumerate(rows)
                        if context_to_id[candidate["Context"].strip()] == int(order[0])
                    )]["Reference"].strip(),
                })
        outputs[name] = {
            "overall": _metrics(ranks),
            "slices": {group: _metrics(values) for group, values in sorted(groups.items())},
            "failures_over_5": sorted(failures, key=lambda item: item["rank"], reverse=True)[:20],
        }

    outputs["dataset"] = {
        "questions": len(rows),
        "unique_contexts": len(contexts),
        "unique_references": len({row["Reference"].strip() for row in rows}),
        "missing_type": sum(not str(row.get("Type", "")).strip() for row in rows),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("evals/RAGCare-QA.json"))
    parser.add_argument("--output", type=Path, default=Path("evals/ragcare_results.json"))
    args = parser.parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = evaluate(rows)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: value["overall"] for name, value in results.items() if "overall" in value}, indent=2))
    print(f"Full report: {args.output}")


if __name__ == "__main__":
    main()
