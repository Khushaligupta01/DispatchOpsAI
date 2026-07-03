"""
tests/unit/test_transcription_service.py

Unit tests for TranscriptionService.

Tests the orchestration logic in complete isolation:
- No HTTP layer
- No real Whisper model (WhisperService is mocked)
- No real filesystem for job lookup (InMemoryJobRepository used directly)
- Real filesystem only for the audio file that the service checks exists

What we test:
- Successful transcription updates all job fields correctly.
- Status transitions: UPLOADED → TRANSCRIBING → TRANSCRIBED.
- JobNotFoundError raised when job_id doesn't exist.
- AudioFileNotFoundError raised when audio file is missing from disk.
- TranscriptionError propagates when Whisper fails, and job is marked FAILED.
- Repository is updated atomically with all transcription fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.repositories.job_repository import InMemoryJobRepository
from app.schemas.job import Job, JobStatus
from app.services.transcription_service import (
    AudioFileNotFoundError,
    JobNotFoundError,
    TranscriptionService,
)
from app.transcription.whisper_service import TranscriptionResult
from app.utils.exceptions import TranscriptionError


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def make_job(job_id: str, file_path: str) -> Job:
    """Create a minimal Job in UPLOADED status pointing to file_path."""
    now = datetime.now(tz=timezone.utc)
    return Job(
        job_id=job_id,
        filename=f"{job_id}.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=2048,
        file_path=file_path,
        status=JobStatus.UPLOADED,
        uploaded_at=now,
        updated_at=now,
    )


def make_mock_whisper(
    transcript: str = "Pipe burst under the sink.",
    duration: float = 8.5,
) -> MagicMock:
    """Build a WhisperService mock that returns a fixed TranscriptionResult."""
    mock = MagicMock()
    mock.transcribe.return_value = TranscriptionResult(
        transcript=transcript,
        duration_seconds=duration,
    )
    return mock


@pytest.fixture
def repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def mock_whisper() -> MagicMock:
    return make_mock_whisper()


@pytest.fixture
def service(
    repo: InMemoryJobRepository, mock_whisper: MagicMock
) -> TranscriptionService:
    return TranscriptionService(repository=repo, whisper_service=mock_whisper)


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_returns_job_with_transcript(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """Successful transcription returns a Job with the transcript text."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    job = make_job("job-001", str(audio_file))
    await repo.save(job)

    result = await service.transcribe_job("job-001")

    assert result.transcript == "Pipe burst under the sink."


@pytest.mark.asyncio
async def test_transcribe_returns_job_with_duration(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """Successful transcription returns a Job with duration_seconds."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    await repo.save(make_job("job-002", str(audio_file)))
    result = await service.transcribe_job("job-002")

    assert result.duration_seconds == 8.5


@pytest.mark.asyncio
async def test_transcribe_status_is_transcribed(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """Successful transcription sets job status to TRANSCRIBED."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    await repo.save(make_job("job-003", str(audio_file)))
    result = await service.transcribe_job("job-003")

    assert result.status == JobStatus.TRANSCRIBED


@pytest.mark.asyncio
async def test_transcribe_sets_transcribed_at(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """Successful transcription populates transcribed_at timestamp."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    await repo.save(make_job("job-004", str(audio_file)))
    result = await service.transcribe_job("job-004")

    assert result.transcribed_at is not None
    assert isinstance(result.transcribed_at, datetime)


@pytest.mark.asyncio
async def test_transcribe_persists_to_repository(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """After transcription, the updated job is retrievable from the repository."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    await repo.save(make_job("job-005", str(audio_file)))
    await service.transcribe_job("job-005")

    stored = await repo.get_by_id("job-005")
    assert stored is not None
    assert stored.status == JobStatus.TRANSCRIBED
    assert stored.transcript == "Pipe burst under the sink."


@pytest.mark.asyncio
async def test_transcribe_status_transitions_through_transcribing(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
    mock_whisper: MagicMock,
):
    """
    Verifies the intermediate TRANSCRIBING status is set before Whisper runs.

    We record the job's status at the moment Whisper is called by inspecting
    the repository inside an async side-effect on the mock.
    """
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"fake audio data")

    await repo.save(make_job("job-006", str(audio_file)))

    # Will hold the status observed mid-pipeline
    observed_status: list = []

    def capture_and_transcribe(path: str) -> TranscriptionResult:
        """
        Synchronous side-effect: reads the job's current status from the
        repo's internal store directly (no async needed here) and records it.
        """
        job = repo._store.get("job-006")
        if job:
            observed_status.append(job.status)
        return TranscriptionResult(transcript="Test", duration_seconds=1.0)

    mock_whisper.transcribe.side_effect = capture_and_transcribe

    result = await service.transcribe_job("job-006")

    # The status observed when Whisper was called must have been TRANSCRIBING
    assert JobStatus.TRANSCRIBING in observed_status, (
        f"Expected TRANSCRIBING during pipeline, got: {observed_status}"
    )
    # Final status must be TRANSCRIBED
    assert result.status == JobStatus.TRANSCRIBED


@pytest.mark.asyncio
async def test_whisper_called_with_correct_file_path(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
    mock_whisper: MagicMock,
):
    """Whisper is called with the exact file_path stored on the job."""
    audio_file = tmp_path / "unique_call.wav"
    audio_file.write_bytes(b"audio")

    await repo.save(make_job("job-007", str(audio_file)))
    await service.transcribe_job("job-007")

    mock_whisper.transcribe.assert_called_once_with(str(audio_file))


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_job_not_found_for_missing_job(
    service: TranscriptionService,
):
    """transcribe_job raises JobNotFoundError for an unknown job_id."""
    with pytest.raises(JobNotFoundError) as exc_info:
        await service.transcribe_job("does-not-exist")

    assert "does-not-exist" in exc_info.value.message


@pytest.mark.asyncio
async def test_raises_audio_file_not_found_when_file_missing(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
):
    """transcribe_job raises AudioFileNotFoundError when file_path doesn't exist."""
    job = make_job("job-008", "/nonexistent/path/audio.wav")
    await repo.save(job)

    with pytest.raises(AudioFileNotFoundError) as exc_info:
        await service.transcribe_job("job-008")

    assert "job-008" in exc_info.value.message


@pytest.mark.asyncio
async def test_job_marked_failed_when_whisper_raises(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
    mock_whisper: MagicMock,
):
    """When Whisper raises TranscriptionError, the job status is set to FAILED."""
    audio_file = tmp_path / "corrupt.wav"
    audio_file.write_bytes(b"not real audio")

    await repo.save(make_job("job-009", str(audio_file)))

    mock_whisper.transcribe.side_effect = TranscriptionError(
        message="Whisper crashed", detail="ffmpeg error"
    )

    with pytest.raises(TranscriptionError):
        await service.transcribe_job("job-009")

    # Job should now be FAILED, not TRANSCRIBING
    stored = await repo.get_by_id("job-009")
    assert stored is not None
    assert stored.status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_transcription_error_propagates_from_whisper(
    service: TranscriptionService,
    repo: InMemoryJobRepository,
    tmp_path: Path,
    mock_whisper: MagicMock,
):
    """TranscriptionError from Whisper propagates out of transcribe_job."""
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"audio")

    await repo.save(make_job("job-010", str(audio_file)))
    mock_whisper.transcribe.side_effect = TranscriptionError("Whisper failed")

    with pytest.raises(TranscriptionError):
        await service.transcribe_job("job-010")
