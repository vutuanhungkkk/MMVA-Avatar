import os
import io
import asyncio
import importlib
import sys
from fastapi import APIRouter, UploadFile, File, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from backend.schemas.requests import ConfigRequest, RemoveDocumentRequest
from backend.api.dependencies import get_assistant, get_session_id
from backend.services.orchestrator import Orchestrator
from assistant.config import RAG_DOCUMENTS_DIR
from backend.security import read_upload_limited, storage_name

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}

router = APIRouter()

@router.post("/api/config")
async def configure_assistant(config: ConfigRequest, request: Request, session_id: str = Depends(get_session_id)):
    os.environ["LLM_PROVIDER"] = config.provider
    os.environ["TTS_PROVIDER"] = "kokoro"

    try:
        modules_to_reload = [key for key in sys.modules if key.startswith("assistant") or key.startswith("backend.services")]
        for mod_name in modules_to_reload:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass

        orchestrator_class = importlib.import_module("backend.services.orchestrator").Orchestrator
        request.app.state.sessions.create_or_replace(session_id, factory=orchestrator_class)
        return JSONResponse(content={"message": "Assistant initialised successfully!", "session_id": session_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/api/clear")
async def clear_conversation(assistant: Orchestrator = Depends(get_assistant)):
    if assistant:
        assistant.reset_conversation()
    return JSONResponse(content={"message": "Conversation cleared."})

import time
import wave
from assistant.utils import log

@router.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), assistant: Orchestrator = Depends(get_assistant)):
    try:
        if not assistant:
            return JSONResponse(status_code=400, content={"error": "Please start assistant first."})

        audio_bytes = await read_upload_limited(audio)
        audio_buffer = io.BytesIO(audio_bytes)

        # Calculate audio duration for RTF
        audio_duration = 0.0
        try:
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                audio_duration = frames / float(rate)
        except Exception:
            pass

        start_time = time.time()
        transcript = assistant.voice_service.transcribe(audio_buffer)
        stt_time = time.time() - start_time

        rtf = stt_time / audio_duration if audio_duration > 0 else 0
        log(f"STT Time: {stt_time:.3f}s | Audio Duration: {audio_duration:.3f}s | RTF: {rtf:.3f}", title="Metrics: STT", style="bold cyan")

        return JSONResponse(content={"transcript": transcript.strip()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/api/process_pdf")
@router.post("/api/process_document")
async def process_pdf(
    request: Request,
    pdf: UploadFile = File(...),
    assistant: Orchestrator = Depends(get_assistant),
    session_id: str = Depends(get_session_id),
):
    if not assistant:
        return JSONResponse(status_code=400, content={"error": "Please start assistant first."})
    try:
        stored_name = storage_name(pdf.filename or "", SUPPORTED_DOCUMENT_EXTENSIONS)

        os.makedirs(RAG_DOCUMENTS_DIR, exist_ok=True)
        file_path = os.path.join(RAG_DOCUMENTS_DIR, stored_name)
        with open(file_path, "wb") as f:
            f.write(await read_upload_limited(pdf))

        await asyncio.to_thread(assistant.rebuild_rag)
        request.app.state.sessions.register_document(session_id, stored_name)
        return JSONResponse(content={"message": f"Successfully indexed {pdf.filename}", "document_id": stored_name})
    except HTTPException:
        raise
    except Exception:
        return JSONResponse(status_code=500, content={"error": "Document processing failed"})

@router.post("/api/remove_document")
async def remove_document(
    req: RemoveDocumentRequest,
    request: Request,
    assistant: Orchestrator = Depends(get_assistant),
    session_id: str = Depends(get_session_id),
):
    if not assistant:
        return JSONResponse(status_code=400, content={"error": "Please start assistant first."})
    try:
        safe_name = os.path.basename(req.filename)
        if not request.app.state.sessions.owns_document(session_id, safe_name):
            raise HTTPException(status_code=404, detail="Document not found")
        file_path = os.path.join(RAG_DOCUMENTS_DIR, safe_name)

        if os.path.isfile(file_path):
            os.remove(file_path)
        request.app.state.sessions.unregister_document(session_id, safe_name)

        await asyncio.to_thread(assistant.rebuild_rag)

        if assistant.rag_service._vectorstore_count() == 0:
            assistant.rag_service.vectorstore = None

        return JSONResponse(content={"message": f"Document '{safe_name}' removed and index updated."})
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
