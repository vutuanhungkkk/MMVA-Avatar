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

app = FastAPI(title="Voice Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize application state for dependency injection
app.state.assistant = None

# Include modularized routers
app.include_router(api_router)
app.include_router(ws_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
