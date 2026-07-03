"""
tests/unit/test_whisper_service.py

Unit tests for WhisperService.

ALL WHISPER MODEL LOADING IS MOCKED.
Tests never download weights, never require ffmpeg, never touch real audio.

What we test here:
- The service calls whisper.load_model() with the correct model size.
- transcribe() returns a TranscriptionResult with the expected fields.
- transcribe() raises TranscriptionError for a missing file.
- transcribe() raises TranscriptionError when Whisper itself fails.
- Duration is extracted from the last segment's 'end' value.
- Duration defaults to 0.0 when segments are empty (silent audio).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.transcription.whisper_service import TranscriptionResult, WhisperService
from app.utils.exceptions import TranscriptionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_whisper_result(
    text: str = "Hello, my HVAC unit is making a loud noise.",
    segments: list | None = None,
) -> dict:
    """Build a fake Whisper result dict matching the real API shape."""
    if segments is None:
        segments = [
            {"start": 0.0, "end": 3.5, "text": "Hello,"},
            {"start": 3.5, "end": 7.2, "text": "my HVAC unit is making a loud noise."},
        ]
    return {"text": text, "segments": segments}


def make_whisper_service(model_size: str = "base") -> WhisperService:
    """
    Instantiate WhisperService with the Whisper model mocked out.

    Patches whisper.load_model so no download or GPU memory allocation occurs.
    Returns both the service and the mock model for assertion use.
    """
    mock_model = MagicMock()
    with patch("app.transcription.whisper_service.whisper") as mock_whisper_module:
        mock_whisper_module.load_model.return_value = mock_model
        service = WhisperService(model_size=model_size)
    # Attach the mock model directly so tests can configure it
    service._model = mock_model
    return service


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------

def test_load_model_called_with_correct_size():
    """WhisperService loads the model with the specified model_size."""
    mock_model = MagicMock()
    with patch("app.transcription.whisper_service.whisper") as mock_whisper:
        mock_whisper.load_model.return_value = mock_model
        WhisperService(model_size="small")
        mock_whisper.load_model.assert_called_once_with("small")


def test_load_model_failure_raises_transcription_error():
    """If whisper.load_model raises, TranscriptionError is raised."""
    with patch("app.transcription.whisper_service.whisper") as mock_whisper:
        mock_whisper.load_model.side_effect = RuntimeError("CUDA out of memory")
        with pytest.raises(TranscriptionError) as exc_info:
            WhisperService(model_size="large")
        assert "large" in exc_info.value.message


# ---------------------------------------------------------------------------
# Successful transcription tests
# ---------------------------------------------------------------------------

def test_transcribe_returns_transcript_text(tmp_path: Path):
    """transcribe() returns the text from the Whisper result."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result(
        text="  My boiler stopped working.  "
    )

    result = service.transcribe(audio_file)

    # Text should be stripped of leading/trailing whitespace
    assert result.transcript == "My boiler stopped working."


def test_transcribe_returns_correct_duration(tmp_path: Path):
    """transcribe() extracts duration from the last segment's 'end' value."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result(
        segments=[
            {"start": 0.0, "end": 4.0, "text": "Pipe is leaking"},
            {"start": 4.0, "end": 9.75, "text": "in the basement"},
        ]
    )

    result = service.transcribe(audio_file)

    assert result.duration_seconds == 9.75


def test_transcribe_duration_zero_for_empty_segments(tmp_path: Path):
    """Duration defaults to 0.0 when Whisper returns no segments (silent audio)."""
    audio_file = tmp_path / "silent.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result(
        text="", segments=[]
    )

    result = service.transcribe(audio_file)

    assert result.duration_seconds == 0.0


def test_transcribe_returns_transcription_result_type(tmp_path: Path):
    """transcribe() returns a TranscriptionResult instance."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result()

    result = service.transcribe(audio_file)

    assert isinstance(result, TranscriptionResult)


def test_transcribe_accepts_path_object(tmp_path: Path):
    """transcribe() accepts a Path object, not just a string."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result()

    # Pass a Path object
    result = service.transcribe(audio_file)

    assert isinstance(result, TranscriptionResult)


def test_transcribe_accepts_string_path(tmp_path: Path):
    """transcribe() accepts a plain string path."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio")

    service = make_whisper_service()
    service._model.transcribe.return_value = make_whisper_result()

    # Pass a string
    result = service.transcribe(str(audio_file))

    assert isinstance(result, TranscriptionResult)


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

def test_transcribe_raises_for_missing_file():
    """transcribe() raises TranscriptionError when the audio file doesn't exist."""
    service = make_whisper_service()

    with pytest.raises(TranscriptionError) as exc_info:
        service.transcribe("/nonexistent/path/audio.wav")

    assert "not found" in exc_info.value.message.lower()


def test_transcribe_raises_transcription_error_on_whisper_failure(tmp_path: Path):
    """If Whisper's model.transcribe() raises, TranscriptionError is raised."""
    audio_file = tmp_path / "corrupt.wav"
    audio_file.write_bytes(b"not real audio")

    service = make_whisper_service()
    service._model.transcribe.side_effect = RuntimeError("ffmpeg decode failed")

    with pytest.raises(TranscriptionError) as exc_info:
        service.transcribe(audio_file)

    assert "ffmpeg decode failed" in exc_info.value.detail
