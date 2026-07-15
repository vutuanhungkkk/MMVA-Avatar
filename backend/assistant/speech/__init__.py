"""Speech recognition and audio utilities with lazy optional audio dependencies."""

from .recognition import wav_to_text, extract_prompt, get_whisper_model


def play_wav_file(audio_path):
    from .audio import play_wav_file as _play_wav_file
    return _play_wav_file(audio_path)


__all__ = [
    "wav_to_text",
    "extract_prompt",
    "get_whisper_model",
    "play_wav_file",
]