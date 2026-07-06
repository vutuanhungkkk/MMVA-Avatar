import asyncio
import base64
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.orchestrator import Orchestrator

ws_router = APIRouter()

@ws_router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    
    assistant: Orchestrator = websocket.app.state.assistant
    
    if not assistant:
        await websocket.send_json({"type": "text_stream", "content": "⚠️ System: Please select a provider and click 'Apply & Start' first."})
        await websocket.send_json({"type": "done", "content": "[DONE]"})

    try:
        while True:
            data = await websocket.receive_json()
            
            assistant = websocket.app.state.assistant
            if not assistant:
                await websocket.send_json({"type": "text_stream", "content": "⚠️ System: Please select a provider and click 'Apply & Start' first."})
                await websocket.send_json({"type": "done", "content": "[DONE]"})
                continue
                
            prompt = data.get("prompt", "")
            img_b64 = data.get("image_b64")
            document_meta = data.get("document")

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
                            await _process_and_send_audio(websocket, assistant, sentence_buffer)
                        break 
                        
                    sentence_buffer += chunk
                    await websocket.send_json({"type": "text_stream", "content": chunk})

                except Exception as e:
                    print(f"LLM Stream Error: {e}")
                    break
            
            await websocket.send_json({"type": "done", "content": "[DONE]"})
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "close message has been sent" in str(e) or "Cannot call" in str(e):
            pass
        else:
            print(f"WebSocket RuntimeError: {e}")

async def _process_and_send_audio(websocket, assistant: Orchestrator, text: str):
    clean_text = text.strip()
    if not clean_text: return
    
    try:
        audio_bytes = assistant.voice_service.get_audio_bytes(clean_text) 
        
        if audio_bytes:
            print(f"Audio bytes length: {len(audio_bytes)}")
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            await websocket.send_json({
                "type": "audio_sentence",
                "text": clean_text,
                "audio_b64": audio_b64
            })
    except Exception as e:
        print(f"TTS generation error for text '{clean_text}': {e}")
