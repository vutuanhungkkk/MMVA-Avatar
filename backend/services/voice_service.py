import importlib
import io
from typing import Any
from assistant.providers.tts import get_tts_provider, TTSProvider
from assistant.speech import wav_to_text
from assistant.config import TTS_PROVIDER
from assistant.utils import log

class VoiceService:
    def __init__(self):
        self.tts_provider: TTSProvider = get_tts_provider()

    def speak(self, text: str) -> None:
        """Speak text using the configured TTS provider with fallback to OpenAI."""
        if self.tts_provider.speak(text):
            return

        if TTS_PROVIDER != "openai":
            log("Falling back to OpenAI TTS.", title="TTS", style="bold yellow")
            try:
                OpenAITTSProvider = importlib.import_module("assistant.providers.tts.openai_tts").OpenAITTSProvider
                fallback = OpenAITTSProvider()
                if fallback.speak(text):
                    return
            except Exception:
                pass

        log("Unable to synthesise speech for the assistant response.", title="TTS", style="bold red")

    def stream_speak(self, text_generator):
        """Stream speak chunks."""
        self.tts_provider.stream_speak(text_generator)
        
    def get_audio_bytes(self, text: str) -> bytes:
        """Get audio bytes for a given text."""
        return self.tts_provider.get_audio_bytes(text)

    def transcribe(self, audio_wav: io.BytesIO) -> str:
        """Transcribe audio WAV using configured STT."""
        return wav_to_text(audio_wav)
