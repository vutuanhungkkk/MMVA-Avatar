import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssl

if hasattr(ssl.SSLContext, '_load_windows_store_certs'):
    ssl.SSLContext._load_windows_store_certs = lambda *args, **kwargs: None

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from backend.api.routes import router as api_router
from backend.api.websockets import ws_router
from backend.services.session_manager import SessionManager
from backend.observability import RequestContextMiddleware

app = FastAPI(title="Voice Assistant API")
app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize application state for dependency injection
app.state.sessions = SessionManager()

@app.get("/health/live")
async def health_live():
    return {"status": "ok"}

@app.get("/health/ready")
async def health_ready():
    return {"status": "ready", "active_sessions": app.state.sessions.active_count}

# Include modularized routers
app.include_router(api_router)
app.include_router(ws_router)

@app.get("/")
async def root():
    return {"message": "Backend is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
