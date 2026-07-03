"""
app/services/transcription_service.py

Transcription service — orchestrates the full transcription pipeline for one job.

WHAT THIS FILE DOES:
--------------------
TranscriptionService has one public method: transcribe_job().

It does exactly these steps in order:
  1. Fetch the Job from the repository.
  2. Validate the Job exists and the audio file is on disk.
  3. Mark the Job as TRANSCRIBING (so callers can observe progress).
  4. Call WhisperService to transcribe the audio.
  5. Attach the transcript, duration, and timestamp to the Job.
  6. Mark the Job as TRANSCRIBED.
  7. Persist the updated Job.
  8. Return the Job.

WHY DOES THE SERVICE NOT CALL WHISPER DIRECTLY?
------------------------------------------------
TranscriptionService depends on WhisperService via injection. In tests,
we inject a mock WhisperService that returns a fixed TranscriptionResult
without loading any model or reading any file. The service logic (status
transitions, error handling, repository updates) is tested in isolation.

WHY DOES STATUS CHANGE HAPPEN BEFORE AND AFTER WHISPER?
--------------------------------------------------------
Setting status to TRANSCRIBING before calling Whisper serves two purposes:
1. If Whisper takes 10 seconds, a polling client sees TRANSCRIBING, not
   UPLOADED, which would look like nothing happened.
2. If the process crashes mid-transcription, the status in the repository
   shows TRANSCRIBING, not UPLOADED — which helps diagnose stuck jobs.

When Celery is added in a later feature, these status transitions will
happen inside the task, making progress visible across the async pipeline.

Interview talking point:
"The service advances the job status before AND after the AI step.
TRANSCRIBING means 'in progress', TRANSCRIBED means 'done'.
If the process crashes between those two states, the stuck status is
observable in the repository — not hidden behind a misleading UPLOADED."
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.repositories.job_repository import AbstractJobRepository
from app.schemas.job import Job, JobStatus
from app.transcription.whisper_service import WhisperService
from app.utils.exceptions import DispatchOpsError, TranscriptionError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class JobNotFoundError(DispatchOpsError):
    """Raised when a job_id does not exist in the repository."""
    pass


class AudioFileNotFoundError(DispatchOpsError):
    """Raised when the audio file for a job is missing from disk."""
    pass


class TranscriptionService:
    """
    Orchestrates the transcription pipeline for a single job.

    Dependencies are injected so they can be swapped in tests.

    Args:
        repository:      Where to load and persist Job objects.
        whisper_service: The speech-to-text service to call.
    """

    def __init__(
        self,
        repository: AbstractJobRepository,
        whisper_service: WhisperService,
    ) -> None:
        self._repo = repository
        self._whisper = whisper_service

    async def transcribe_job(self, job_id: str) -> Job:
        """
        Run the transcription pipeline for the job identified by job_id.

        Args:
            job_id: The UUID string of the job to transcribe.

        Returns:
            The fully updated Job with status=TRANSCRIBED, transcript,
            duration_seconds, and transcribed_at populated.

        Raises:
            JobNotFoundError:       No job with this job_id exists.
            AudioFileNotFoundError: The audio file is missing from disk.
            TranscriptionError:     Whisper failed to process the audio.
        """
        logger.info(
            "Transcription pipeline started",
            extra={"job_id": job_id},
        )

        # Step 1 — Fetch the job
        job = await self._repo.get_by_id(job_id)
        if job is None:
            logger.warning(
                "Transcription failed — job not found",
                extra={"job_id": job_id},
            )
            raise JobNotFoundError(
                message=f"Job '{job_id}' not found.",
                detail="Ensure the job was created via POST /api/v1/jobs/upload-audio first.",
            )

        # Step 2 — Validate the audio file exists on disk
        from pathlib import Path
        if not Path(job.file_path).exists():
            logger.error(
                "Transcription failed — audio file missing from disk",
                extra={
                    "job_id": job_id,
                    "file_path": job.file_path,
                },
            )
            raise AudioFileNotFoundError(
                message=f"Audio file not found for job '{job_id}'.",
                detail=f"Expected file at: '{job.file_path}'. It may have been deleted.",
            )

        # Step 3 — Mark as TRANSCRIBING so progress is visible
        await self._repo.update_status(job_id, JobStatus.TRANSCRIBING)

        logger.info(
            "Transcription started",
            extra={
                "job_id": job_id,
                "audio_filename": job.filename,
                "file_path": job.file_path,
            },
        )

        # Step 4 — Run Whisper (may raise TranscriptionError)
        try:
            result = self._whisper.transcribe(job.file_path)
        except TranscriptionError:
            # Mark job as FAILED before re-raising so the status is accurate
            await self._repo.update_status(job_id, JobStatus.FAILED)
            logger.error(
                "Transcription failed — Whisper error",
                extra={"job_id": job_id, "audio_filename": job.filename},
            )
            raise

        # Step 5 — Build the updated Job with all transcription fields populated
        now = datetime.now(tz=timezone.utc)
        updated_job = job.model_copy(
            update={
                "transcript": result.transcript,
                "duration_seconds": result.duration_seconds,
                "transcribed_at": now,
                "updated_at": now,
                "status": JobStatus.TRANSCRIBED,
            }
        )

        # Step 6 — Persist the full update atomically
        persisted_job = await self._repo.update_job(updated_job)

        logger.info(
            "Transcription completed",
            extra={
                "job_id": job_id,
                "audio_filename": job.filename,
                "duration_seconds": result.duration_seconds,
                "transcript_length": len(result.transcript),
            },
        )

        # update_job only returns None if the job_id disappeared between
        # get_by_id and update_job — extremely unlikely, but guard anyway.
        return persisted_job or updated_job
