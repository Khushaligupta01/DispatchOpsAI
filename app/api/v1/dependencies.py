"""
app/api/v1/dependencies.py

Route-level FastAPI dependency providers.

This file wires concrete implementations to the abstract interfaces
that services depend on. It is the composition root for the API layer.

WHY IS THIS SEPARATE FROM app/dependencies.py?
----------------------------------------------
app/dependencies.py holds application-wide dependencies (DB session, settings).
This file holds feature-specific dependencies (which service and repository
implementations to inject for a given set of routes).

As the project grows, each feature adds its provider here without
touching global application wiring.

WHY A MODULE-LEVEL SINGLETON FOR THE REPOSITORY?
-------------------------------------------------
InMemoryJobRepository stores state in a dict. If we created a new instance
on every request, every request would start with an empty store — uploads
from previous requests would vanish.

Using a module-level instance means all requests share the same dict.
When Feature 5 replaces this with PostgresJobRepository, the singleton
pattern is no longer needed (the database persists state externally),
but the interface stays identical — one line changes here.

WHY A MODULE-LEVEL SINGLETON FOR WhisperService?
-------------------------------------------------
WhisperService loads a ~150 MB model on instantiation. Creating a new
instance per request would reload the model on every transcription call,
adding 2-5 seconds of overhead each time.

The module-level singleton ensures the model loads once at startup and
is reused for the lifetime of the process.

Interview talking point:
"The dependency providers are the only place that knows which concrete
implementations are in use. Swapping in-memory to PostgreSQL, or Whisper
to a cloud STT API, is a one-line change here. The services, routes,
and tests never change."
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.repositories.job_repository import InMemoryJobRepository
from app.services.transcription_service import TranscriptionService
from app.services.upload_service import UploadService
from app.transcription.whisper_service import WhisperService

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------

# All services share this one repository instance — the in-memory store
# that persists job state for the lifetime of the process.
# Replaced with a database-backed repository in Feature 5.
_job_repository = InMemoryJobRepository()


@lru_cache
def _get_whisper_service() -> WhisperService:
    """
    Load and cache the WhisperService singleton.

    The @lru_cache ensures the Whisper model is loaded exactly once,
    regardless of how many requests arrive. This function is intentionally
    private — external code uses get_transcription_service() instead.
    """
    settings = get_settings()
    return WhisperService(model_size=settings.whisper_model_size)


# ---------------------------------------------------------------------------
# Public dependency providers
# ---------------------------------------------------------------------------

@lru_cache
def get_upload_service() -> UploadService:
    """
    Dependency provider for UploadService.

    FastAPI injects this into any route that declares:
        upload_service: UploadService = Depends(get_upload_service)
    """
    settings = get_settings()
    return UploadService(
        repository=_job_repository,
        upload_dir=settings.audio_upload_dir,
    )


def get_transcription_service() -> TranscriptionService:
    """
    Dependency provider for TranscriptionService.

    Returns a TranscriptionService backed by:
    - The shared in-memory repository (same instance as upload service).
    - The cached WhisperService singleton (model loaded once).

    NOT cached with @lru_cache because TranscriptionService itself holds
    no state — its dependencies (repository and whisper) are singletons.
    Creating a lightweight service wrapper per request is negligible cost
    and avoids lru_cache's thread-safety edge cases.

    FastAPI injects this into any route that declares:
        transcription_service: TranscriptionService = Depends(get_transcription_service)
    """
    return TranscriptionService(
        repository=_job_repository,
        whisper_service=_get_whisper_service(),
    )
