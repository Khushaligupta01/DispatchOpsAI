"""
app/utils/exceptions.py

Custom exception hierarchy for DispatchOps AI.

Why a custom exception hierarchy?
- It makes error handling explicit and intentional.
- Each layer of the app raises a specific exception type, not a generic Exception.
- Celery tasks can catch specific exceptions and decide:
    - Is this retriable? (network timeout)
    - Is this a permanent failure? (bad audio format)
- FastAPI exception handlers can map specific exceptions to HTTP status codes.
- Langfuse spans can record the exact exception type for observability.

Interview talking point:
"Every error has a type. When a Celery task fails, the exception class tells us
whether to retry, flag for human review, or alert the on-call engineer."
"""

from __future__ import annotations  # Enables X | Y union syntax on Python 3.10


class DispatchOpsError(Exception):
    """
    Base exception for all DispatchOps AI errors.

    All custom exceptions inherit from this so you can catch any
    application-level error with: except DispatchOpsError.
    """

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r})"


# --- Audio / Twilio Layer ---

class AudioDownloadError(DispatchOpsError):
    """Raised when the call recording cannot be downloaded from Twilio."""
    pass


# --- Transcription Layer ---

class TranscriptionError(DispatchOpsError):
    """Raised when Whisper fails to transcribe the audio file."""
    pass


# --- Extraction Layer ---

class ExtractionError(DispatchOpsError):
    """Raised when the LLM fails to extract structured data from the transcript."""
    pass


class LowConfidenceError(DispatchOpsError):
    """
    Raised when the extraction confidence score is below the threshold.

    This is NOT a system error — it means the LLM was uncertain about
    the job details and a human should review the call instead of
    auto-dispatching.
    """
    pass


# --- Job Domain (shared across pipeline services) ---

class JobNotFoundError(DispatchOpsError):
    """
    Raised when a job_id does not exist in the repository.

    Defined here (not in a service module) because multiple services
    raise this error — defining it once avoids duplication and circular imports.
    """
    pass


# --- Ranking Layer ---

class RankingError(DispatchOpsError):
    """Raised when no available technicians match the job requirements."""
    pass


# --- Dispatch Layer ---

class DispatchError(DispatchOpsError):
    """Raised when the dispatch assignment cannot be saved or sent."""
    pass