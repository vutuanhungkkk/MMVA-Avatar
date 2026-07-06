import pytest
from pathlib import Path
from backend.services.rag_service import RAGService

@pytest.fixture
def rag_service():
    # RAG_ENABLED might be false in tests if config is not patched, 
    # but we can still test the helper methods.
    return RAGService()

def test_chunk_text(rag_service):
    text = "A" * 100
    # chunk size 40, overlap 10
    chunks = rag_service._chunk_text(text, 40, 10)
    assert len(chunks) == 4
    assert chunks[0] == "A" * 40
    # Overlap is 10, step is 30.
    # 0: 0-40
    # 1: 30-70
    # 2: 60-100
    # 3: 90-130 (which is 90-100) -> 10 chars
    assert len(chunks[3]) == 10

def test_supported_document_suffixes(rag_service):
    suffixes = rag_service._supported_document_suffixes()
    assert ".pdf" in suffixes
    assert ".txt" in suffixes
    assert ".csv" in suffixes

def test_read_document_text_txt(rag_service, tmp_path):
    # Create a temporary txt file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, this is a test.")

    content = rag_service._read_document_text(test_file)
    assert content == "Hello, this is a test."
