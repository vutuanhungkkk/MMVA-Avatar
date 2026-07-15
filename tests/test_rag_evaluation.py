from evals.run_rag_eval import evaluate


def test_retrieval_metrics():
    rows = [{"expected_source": "record.pdf", "retrieved": [
        {"source": "other.md"}, {"source": "record.pdf"}
    ]}]
    assert evaluate(rows, k=1)["recall@1"] == 0.0
    assert evaluate(rows, k=5)["recall@5"] == 1.0
    assert evaluate(rows)["mrr"] == 0.5
