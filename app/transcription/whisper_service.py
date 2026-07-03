"""
app/transcription/whisper_service.py

Whisper speech-to-text service for DispatchOps AI.

One class. One responsibility: receive an audio file path, return a transcript
and the audio duration.

WHY IS whisper IMPORTED AT MODULE LEVEL?
-----------------------------------------
Importing whisper at module level (inside a try/except) means:
- unittest.mock.patch("app.transcription.whisper_service.whisper") works.
- Tests replace the module-level name with a MagicMock before WhisperService
  is instantiated — no model download, no ffmpeg required during testing.
- If whisper is not installed, the ImportError is caught gracefully and
  whisper is set to None, so the module is always importable.

WHY LOAD THE MODEL ONCE?
-------------------------
Loading a Whisper model takes 2-5 seconds and ~150 MB of memory.
Loading it per-request would make every transcription 2-5s slower.
WhisperService holds self._model for the process lifetime.
FastAPI's dependency injection returns the same instance on every request.

Interview talking point:
"Whisper loads once at service instantiation. The model stays in memory
for the process lifetime. A cold start costs 3 seconds. After that, each
transcription only costs inference time — roughly 1-5x the audio duration."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.utils.exceptions import TranscriptionError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level import so tests can patch this name via:
#   patch("app.transcription.whisper_service.whisper")
# If openai-whisper is not installed, whisper is None and WhisperService
try:
    import whisper
    print("✅ Whisper imported successfully")
except Exception as e:
    print("❌ Whisper import failed:", repr(e))
    raise


@dataclass
class TranscriptionResult:
    """
    Value object returned by WhisperService.transcribe().

    Using a dataclass (not a plain dict) gives callers IDE autocompletion
    and makes the shape self-documenting.

    Attributes:
        transcript:       Full text of the transcribed audio.
        duration_seconds: Duration of the audio clip in seconds.
    """

    transcript: str
    duration_seconds: float


class WhisperService:
    """
    Wraps the OpenAI Whisper model for local speech-to-text transcription.

    The model is loaded once in __init__ and reused for every transcription.
    Public API: transcribe(audio_path) -> TranscriptionResult.

    Args:
        model_size: Whisper model variant.
                    Options: tiny | base | small | medium | large
                    'base' is the default — balances speed and accuracy.
    """

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size
        self._model = self._load_model(model_size)

    def _load_model(self, model_size: str):  # type: ignore[return]
        """
        Load the Whisper model into memory. Called once at instantiation.

        Raises:
            TranscriptionError: If whisper is not installed or load fails.
        """
        if whisper is None:
            raise TranscriptionError(
                message="openai-whisper is not installed.",
                detail="Run: pip install openai-whisper",
            )
        try:
            logger.info("Loading Whisper model", extra={"model_size": model_size})
            model = whisper.load_model(model_size)
            logger.info("Whisper model loaded", extra={"model_size": model_size})
            return model
        except Exception as exc:
            raise TranscriptionError(
                message=f"Failed to load Whisper model '{model_size}'.",
                detail=str(exc),
            ) from exc

    def transcribe(self, audio_path: str | Path) -> TranscriptionResult:
        """
        Transcribe an audio file and return text + duration.

        Args:
            audio_path: Path to the audio file (WAV, MP3, M4A, or any
                        format supported by ffmpeg).

        Returns:
            TranscriptionResult(transcript, duration_seconds)

        Raises:
            TranscriptionError: File missing, unreadable, or Whisper fails.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise TranscriptionError(
                message=f"Audio file not found: '{audio_path}'",
                detail="The file may have been deleted or the path is incorrect.",
            )

        logger.info(
            "Starting Whisper transcription",
            extra={"audio_path": str(audio_path), "model_size": self._model_size},
        )

        try:
            # Whisper returns {"text": str, "segments": [...], ...}
            result = self._model.transcribe(str(audio_path))
            transcript = result["text"].strip()

            # Duration = end time of the last segment.
            # Falls back to 0.0 for silent or empty audio.
            segments = result.get("segments", [])
            duration = float(segments[-1]["end"]) if segments else 0.0

            logger.info(
                "Whisper transcription completed",
                extra={
                    "audio_path": str(audio_path),
                    "duration_seconds": duration,
                    "transcript_length": len(transcript),
                },
            )
            return TranscriptionResult(transcript=transcript, duration_seconds=duration)

        except TranscriptionError:
            raise  # Re-raise without wrapping
        except Exception as exc:
            logger.error(
                "Whisper transcription failed",
                extra={
                    "audio_path": str(audio_path),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            raise TranscriptionError(
                message="Whisper failed to transcribe the audio file.",
                detail=str(exc),
            ) from exc
