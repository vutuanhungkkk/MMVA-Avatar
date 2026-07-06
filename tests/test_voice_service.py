import pytest
from unittest.mock import patch, MagicMock
from backend.services.voice_service import VoiceService
import io

@pytest.fixture
def mock_get_tts_provider():
    with patch("backend.services.voice_service.get_tts_provider") as mock:
        yield mock

@pytest.fixture
def mock_wav_to_text():
    with patch("backend.services.voice_service.wav_to_text") as mock:
        yield mock

def test_voice_service_transcribe(mock_get_tts_provider, mock_wav_to_text):
    mock_wav_to_text.return_value = "hello world"
    
    service = VoiceService()
    
    dummy_audio = io.BytesIO(b"dummy_audio_bytes")
    transcript = service.transcribe(dummy_audio)
    
    assert transcript == "hello world"
    mock_wav_to_text.assert_called_once_with(dummy_audio)

def test_voice_service_get_audio_bytes(mock_get_tts_provider):
    mock_tts = MagicMock()
    mock_tts.get_audio_bytes.return_value = b"audio_bytes"
    mock_get_tts_provider.return_value = mock_tts
    
    service = VoiceService()
    audio = service.get_audio_bytes("hello world")
    
    assert audio == b"audio_bytes"
    mock_tts.get_audio_bytes.assert_called_once_with("hello world")
