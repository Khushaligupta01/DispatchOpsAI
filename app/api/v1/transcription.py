"""
app/api/v1/transcription.py

Transcription route for DispatchOps AI.

Single endpoint:
    POST /api/v1/jobs/{job_id}/transcribe

This route is intentionally minimal:
- Extract the job_id from the path.
- Call TranscriptionService.
- Map domain exceptions to HTTP status codes.
- Return a TranscriptionResponse.

Zero business logic lives here.

WHY POST AND NOT GET?
---------------------
Transcription is a side-effecting operation — it changes the job's status,
writes a transcript to the repository, and consumes compute time. Operations
that mutate state use POST (or PUT/PATCH). GET is reserved for read-only
retrieval. A client that accidentally calls this endpoint twice would trigger
two transcriptions; using POST makes that intentional behavior clear.

Interview talking point:
"This is a POST because transcription is a state-mutating operation.
GET is idempotent — calling it multiple times has no side effects.
POST is not — each call triggers Whisper inference and advances the
job's status. Using the correct HTTP verb communicates that intent."
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.job import TranscriptionResponse
from app.services.transcription_service import (
    AudioFileNotFoundError,
    JobNotFoundError,
    TranscriptionService,
)
from app.utils.exceptions import TranscriptionError
from app.utils.logger import get_logger

from .dependencies import get_transcription_service

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["Transcription"])


@router.post(
    "/{job_id}/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    summary="Transcribe a job's audio recording",
    description=(
        "Loads the audio file for the given job, runs Whisper transcription, "
        "and stores the transcript on the job. "
        "Returns the transcript and audio duration on success."
    ),
    responses={
        200: {"description": "Transcription completed successfully"},
        400: {"description": "Audio file is missing from disk"},
        404: {"description": "Job not found"},
        500: {"description": "Whisper transcription failed"},
    },
)
async def transcribe_job(
    job_id: str,
    transcription_service: TranscriptionService = Depends(get_transcription_service),
) -> TranscriptionResponse:
    """
    Transcribe the audio recording for the specified job.

    Advances the job status from UPLOADED → TRANSCRIBING → TRANSCRIBED.
    Returns the completed transcript and audio duration.
    """
    logger.info(
        "Transcribe request received",
        extra={"job_id": job_id},
    )

    try:
        job = await transcription_service.transcribe_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc
    except AudioFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except TranscriptionError as exc:
        logger.error(
            "Transcription endpoint — Whisper error",
            extra={"job_id": job_id, "error": exc.message},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.error(
            "Transcription endpoint — unexpected error",
            extra={
                "job_id": job_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during transcription.",
        ) from exc

    # These fields are guaranteed non-None after a successful transcription.
    # The type: ignore comments suppress mypy's Optional warnings here —
    # the service contract guarantees they are populated on TRANSCRIBED jobs.
    return TranscriptionResponse(
        job_id=job.job_id,
        status=job.status,
        transcript=job.transcript,  # type: ignore[arg-type]
        duration_seconds=job.duration_seconds,  # type: ignore[arg-type]
        transcribed_at=job.transcribed_at,  # type: ignore[arg-type]
        message="Transcription completed successfully.",
    )
