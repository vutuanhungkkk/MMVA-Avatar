"""
Health & Medical AI Assistant package.

A multimodal virtual assistant specialized in health & medical support.
Supports DeepSeek, OpenAI, Anthropic Claude, Google Gemini, and local models
with configurable TTS (Kokoro). Features include prescription reading (Vision),
medical document Q&A (RAG), and voice-based symptom description.
"""

# Load .env BEFORE any submodule imports so env-var-driven config picks it up.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional — env vars exported by the shell still work.
    pass

__version__ = "0.3.0"

__all__ = [
    "__version__",
]
