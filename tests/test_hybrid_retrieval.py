from dataclasses import dataclass, field

from backend.services.hybrid_retrieval import reciprocal_rank_fusion


@dataclass
class Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


def test_rrf_rewards_documents_found_by_both_retrievers():
    shared = Document("shared", {"context_id": 1})
    dense_first = Document("dense", {"context_id": 2})
    lexical_first = Document("lexical", {"context_id": 3})

    fused = reciprocal_rank_fusion(
        [[dense_first, shared], [lexical_first, shared]], limit=3, rank_constant=60
    )

    assert fused[0] is shared
    assert fused[0].metadata["rrf_score"] > fused[1].metadata["rrf_score"]


def test_rrf_deduplicates_by_document_identity():
    first = Document("same", {"source": "a.pdf", "chunk": "1"})
    duplicate = Document("same", {"source": "a.pdf", "chunk": "1"})
    assert len(reciprocal_rank_fusion([[first], [duplicate]], limit=5)) == 1
