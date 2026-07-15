import asyncio
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import time

from backend.services.orchestrator import Orchestrator
from assistant.utils import log

ws_router = APIRouter()

active_connections = 0

@ws_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    global active_connections
    await websocket.accept()
    active_connections += 1
    log(f"New connection. Active WS users: {active_connections}", title="Metrics: Concurrency", style="bold green")

    session_id = (websocket.query_params.get("session_id") or "").strip()
    if not session_id or len(session_id) > 128:
        await websocket.close(code=1008, reason="A valid session_id is required")
        active_connections = max(0, active_connections - 1)
        return
    assistant: Orchestrator = websocket.app.state.sessions.get(session_id)

    if not assistant:
        await websocket.send_json({"type": "text_stream", "content": "⚠️ System: Please select a provider and click 'Apply & Start' first."})
        await websocket.send_json({"type": "done", "content": "[DONE]"})

    try:
        while True:
            data = await websocket.receive_json()

            assistant = websocket.app.state.sessions.get(session_id)
            if not assistant:
                await websocket.send_json({"type": "text_stream", "content": "⚠️ System: Please select a provider and click 'Apply & Start' first."})
                await websocket.send_json({"type": "done", "content": "[DONE]"})
                continue

            prompt = data.get("prompt", "")
            img_b64 = data.get("image_b64")
            document_meta = data.get("document")
            if document_meta and not websocket.app.state.sessions.owns_document(session_id, document_meta.get("name", "")):
                await websocket.send_json({"type": "error", "content": "Document does not belong to this session."})
                document_meta = None

            request_start_time = time.time()
            ttft_recorded = False
            token_count = 0
            first_token_time = 0
            last_chunk_time = 0
            metrics_state = {"ttfa_recorded": False}

            def llm_generator():
                for c in assistant.chat_stream_text(prompt, img_b64, document=document_meta):
                    yield c

            loop = asyncio.get_event_loop()
            iterator = iter(llm_generator())

            sentence_buffer = ""

            while True:
                try:
                    chunk = await loop.run_in_executor(None, next, iterator, None)

                    if chunk is None:
                        if sentence_buffer.strip():
                            await _process_and_send_audio(websocket, assistant, sentence_buffer, request_start_time, metrics_state)
                        break

                    current_time = time.time()
                    if not ttft_recorded:
                        ttft = current_time - request_start_time
                        log(f"Time To First Token (TTFT): {ttft*1000:.1f} ms", title="Metrics: LLM", style="bold magenta")
                        ttft_recorded = True
                        first_token_time = current_time
                        last_chunk_time = current_time
                    else:
                        token_count += 1
                        last_chunk_time = current_time

                    sentence_buffer += chunk
                    await websocket.send_json({"type": "text_stream", "content": chunk})

                except Exception as e:
                    print(f"LLM Stream Error: {e}")
                    break

            if token_count > 0:
                itl = ((last_chunk_time - first_token_time) / token_count) * 1000
                speed = 1000 / itl if itl > 0 else 0
                log(f"Inter-Token Latency (ITL): {itl:.1f} ms/token | Gen Speed: {speed:.1f} tokens/s", title="Metrics: LLM", style="bold magenta")

            await websocket.send_json({"type": "done", "content": "[DONE]"})
    except WebSocketDisconnect:
        active_connections = max(0, active_connections - 1)
        log(f"Connection closed. Active WS users: {active_connections}", title="Metrics: Concurrency", style="bold green")
    except RuntimeError as e:
        if "close message has been sent" in str(e) or "Cannot call" in str(e):
            pass
        else:
            print(f"WebSocket RuntimeError: {e}")

async def _process_and_send_audio(websocket, assistant: Orchestrator, text: str, request_start_time: float = 0.0, metrics_state: dict = None):
    clean_text = text.strip()
    if not clean_text: return

    try:
        audio_bytes = assistant.voice_service.get_audio_bytes(clean_text)

        if audio_bytes:
            if request_start_time > 0 and metrics_state and not metrics_state.get("ttfa_recorded", False):
                ttfa = time.time() - request_start_time
                log(f"Time To First Audio (TTFA): {ttfa:.3f} s", title="Metrics: TTS", style="bold magenta")
                metrics_state["ttfa_recorded"] = True

            print(f"Audio bytes length: {len(audio_bytes)}")
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            await websocket.send_json({
                "type": "audio_sentence",
                "text": clean_text,
                "audio_b64": audio_b64
            })
    except Exception as e:
        print(f"TTS generation error for text '{clean_text}': {e}")
