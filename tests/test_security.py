import pytest
from fastapi import HTTPException

from backend.security import storage_name


def test_storage_name_discards_path_and_original_name():
    result = storage_name("../../private/record.pdf", {".pdf"})
    assert result.endswith(".pdf")
    assert "private" not in result
    assert ".." not in result


def test_storage_name_rejects_unapproved_extension():
    with pytest.raises(HTTPException) as error:
        storage_name("payload.exe", {".pdf"})
    assert error.value.status_code == 400
