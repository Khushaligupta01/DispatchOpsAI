"""
tests/integration/test_transcription_api.py

Integration tests for POST /api/v1/jobs/{job_id}/transcribe.

These tests exercise the full HTTP → service → repository chain.
Whisper is mocked — no model download, no ffmpeg, no GPU required.

Strategy:
1. Override get_transcription_service to inject a TranscriptionService
   backed by a mock WhisperService.
2. Pre-populate the repository with a job that has a real audio file on disk.
3. POST to the transcription endpoint.
4. Assert on the HTTP status code and response body.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.dependencies import get_transcription_service, get_upload_service
from app.main import app
from app.repositories.job_repository import InMemoryJobRepository
from app.schemas.job import Job, JobStatus
from app.services.transcription_service import TranscriptionService
from app.services.upload_service import UploadService
from app.transcription.whisper_service import TranscriptionResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_whisper(
    transcript: str = "My air conditioner stopped working.",
    duration: float = 6.3,
) -> MagicMock:
    """Mock WhisperService that returns a fixed result without loading a model."""
    mock = MagicMock()
    mock.transcribe.return_value = TranscriptionResult(
        transcript=transcript,
        duration_seconds=duration,
    )
    return mock


@pytest.fixture
def shared_repo() -> InMemoryJobRepository:
    """Single repository instance shared between upload and transcription services."""
    return InMemoryJobRepository()


@pytest.fixture
def mock_whisper() -> MagicMock:
    return make_mock_whisper()


@pytest.fixture
async def client(
    shared_repo: InMemoryJobRepository,
    mock_whisper: MagicMock,
    tmp_path: Path,
):
    """
    Async test client with both upload and transcription services overridden.

    Both services share the same repository so an uploaded job is visible
    to the transcription service — mirroring production behavior.
    """
    upload_svc = UploadService(repository=shared_repo, upload_dir=str(tmp_path))
    transcription_svc = TranscriptionService(
        repository=shared_repo,
        whisper_service=mock_whisper,
    )

    app.dependency_overrides[get_upload_service] = lambda: upload_svc
    app.dependency_overrides[get_transcription_service] = lambda: transcription_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def upload_audio_file(client: AsyncClient, tmp_path: Path) -> str:
    """
    Helper: upload a real audio file and return the job_id.

    Creates a small WAV file on disk so the transcription service's
    file-existence check passes.
    """
    audio_file = tmp_path / "test_call.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 1020)  # Minimal fake WAV

    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files={"file": ("test_call.wav", io.BytesIO(audio_file.read_bytes()), "audio/wav")},
    )
    assert response.status_code == 201, f"Upload failed: {response.json()}"
    return response.json()["job_id"]


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_returns_200(client: AsyncClient, tmp_path: Path):
    """POST /{job_id}/transcribe returns HTTP 200 on success."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_transcribe_response_contains_transcript(
    client: AsyncClient, tmp_path: Path
):
    """Response body contains the transcript text."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    body = response.json()
    assert body["transcript"] == "My air conditioner stopped working."


@pytest.mark.asyncio
async def test_transcribe_response_status_is_transcribed(
    client: AsyncClient, tmp_path: Path
):
    """Response body shows status=TRANSCRIBED."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert response.json()["status"] == "TRANSCRIBED"


@pytest.mark.asyncio
async def test_transcribe_response_contains_duration(
    client: AsyncClient, tmp_path: Path
):
    """Response body contains duration_seconds."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert response.json()["duration_seconds"] == 6.3


@pytest.mark.asyncio
async def test_transcribe_response_schema(client: AsyncClient, tmp_path: Path):
    """Response body contains all expected fields."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    body = response.json()

    expected_fields = {
        "job_id", "status", "transcript",
        "duration_seconds", "transcribed_at", "message",
    }
    assert expected_fields.issubset(body.keys())


@pytest.mark.asyncio
async def test_transcribe_response_contains_job_id(
    client: AsyncClient, tmp_path: Path
):
    """Response job_id matches the uploaded job."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert response.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_transcribe_response_contains_success_message(
    client: AsyncClient, tmp_path: Path
):
    """Response body contains the success message."""
    job_id = await upload_audio_file(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert response.json()["message"] == "Transcription completed successfully."


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcribe_returns_404_for_unknown_job(client: AsyncClient):
    """POST with an unknown job_id returns HTTP 404."""
    response = await client.post("/api/v1/jobs/nonexistent-job-id/transcribe")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_transcribe_404_contains_detail(client: AsyncClient):
    """404 response includes a detail message."""
    response = await client.post("/api/v1/jobs/ghost-job/transcribe")
    body = response.json()
    assert "detail" in body
    assert len(body["detail"]) > 0


@pytest.mark.asyncio
async def test_transcribe_returns_500_on_whisper_failure(
    shared_repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """POST returns HTTP 500 when Whisper raises TranscriptionError."""
    from app.utils.exceptions import TranscriptionError as TE

    failing_whisper = MagicMock()
    failing_whisper.transcribe.side_effect = TE("Whisper crashed")

    failing_svc = TranscriptionService(
        repository=shared_repo,
        whisper_service=failing_whisper,
    )

    upload_svc = UploadService(repository=shared_repo, upload_dir=str(tmp_path))

    app.dependency_overrides[get_upload_service] = lambda: upload_svc
    app.dependency_overrides[get_transcription_service] = lambda: failing_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        # Upload a real file first
        audio_file = tmp_path / "call.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        upload_resp = await ac.post(
            "/api/v1/jobs/upload-audio",
            files={"file": ("call.wav", io.BytesIO(audio_file.read_bytes()), "audio/wav")},
        )
        job_id = upload_resp.json()["job_id"]

        response = await ac.post(f"/api/v1/jobs/{job_id}/transcribe")

    app.dependency_overrides.clear()

    assert response.status_code == 500
