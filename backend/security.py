"""Input boundaries shared by upload endpoints."""

from pathlib import Path
import re
from uuid import uuid4

from fastapi import HTTPException, UploadFile

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def storage_name(filename: str, allowed_extensions: set[str]) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")
    original = Path((filename or "").replace("\\", "/")).name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original).stem).strip(".-") or "document"
    return f"{uuid4().hex[:12]}-{stem}{suffix}"


async def read_upload_limited(upload: UploadFile, limit: int = MAX_UPLOAD_BYTES) -> bytes:
    chunks = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(status_code=413, detail="Upload exceeds the 20 MB limit")
        chunks.append(chunk)
    return b"".join(chunks)
