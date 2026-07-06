import os
import io
import asyncio
import importlib
import sys
from fastapi import APIRouter, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse

from backend.schemas.requests import ConfigRequest, RemoveDocumentRequest
from backend.api.dependencies import get_assistant
from backend.services.orchestrator import Orchestrator
from assistant.config import RAG_DOCUMENTS_DIR

SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".docx"}

router = APIRouter()

@router.post("/api/config")
async def configure_assistant(config: ConfigRequest, request: Request):
    os.environ["LLM_PROVIDER"] = config.provider
    os.environ["TTS_PROVIDER"] = "kokoro"

    try:
        modules_to_reload = [key for key in sys.modules if key.startswith("assistant") or key.startswith("backend.services")]
        for mod_name in modules_to_reload:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                pass

        request.app.state.assistant = Orchestrator()
        return JSONResponse(content={"message": "Assistant initialised successfully!"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/api/clear")
async def clear_conversation(assistant: Orchestrator = Depends(get_assistant)):
    if assistant:
        assistant.reset_conversation()
    return JSONResponse(content={"message": "Conversation cleared."})

@router.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), assistant: Orchestrator = Depends(get_assistant)):
    try:
        if not assistant:
            return JSONResponse(status_code=400, content={"error": "Please start assistant first."})
        audio_buffer = io.BytesIO(await audio.read())
        transcript = assistant.voice_service.transcribe(audio_buffer)
        return JSONResponse(content={"transcript": transcript.strip()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/api/process_pdf")
@router.post("/api/process_document")
async def process_pdf(pdf: UploadFile = File(...), assistant: Orchestrator = Depends(get_assistant)):
    if not assistant:
        return JSONResponse(status_code=400, content={"error": "Please start assistant first."})
    try:
        ext = os.path.splitext(pdf.filename or "")[1].lower()
        if ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported file type: {ext or 'unknown'}"},
            )

        os.makedirs(RAG_DOCUMENTS_DIR, exist_ok=True)
        file_path = os.path.join(RAG_DOCUMENTS_DIR, pdf.filename)
        with open(file_path, "wb") as f:
            f.write(await pdf.read())
        
        await asyncio.to_thread(assistant.rebuild_rag)
        return JSONResponse(content={"message": f"Successfully indexed {pdf.filename}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/api/remove_document")
async def remove_document(req: RemoveDocumentRequest, assistant: Orchestrator = Depends(get_assistant)):
    if not assistant:
        return JSONResponse(status_code=400, content={"error": "Please start assistant first."})
    try:
        safe_name = os.path.basename(req.filename)
        file_path = os.path.join(RAG_DOCUMENTS_DIR, safe_name)

        if os.path.isfile(file_path):
            os.remove(file_path)
        
        await asyncio.to_thread(assistant.rebuild_rag)

        if assistant.rag_service._vectorstore_count() == 0:
            assistant.rag_service.vectorstore = None

        return JSONResponse(content={"message": f"Document '{safe_name}' removed and index updated."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
