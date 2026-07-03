"""
tests/integration/test_extraction_api.py

Integration tests for POST /api/v1/jobs/{job_id}/extract.

Strategy:
1. Override both get_upload_service and get_extraction_service to inject
   services backed by the same shared in-memory repository.
2. Upload a real file, transcribe it (via mock Whisper), then extract.
3. Assert on HTTP status codes and response body shape.

Groq is never called. The real API is never hit.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.dependencies import (
    get_extraction_service,
    get_transcription_service,
    get_upload_service,
)
from app.llm.groq_client import GroqClient
from app.main import app
from app.repositories.job_repository import InMemoryJobRepository
from app.services.extraction_service import ExtractionService
from app.services.transcription_service import TranscriptionService
from app.services.upload_service import UploadService
from app.transcription.whisper_service import TranscriptionResult
from app.utils.exceptions import ExtractionError


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_EXTRACTION = json.dumps({
    "customer_name": "Maria Garcia",
    "address": "88 Elm Drive, Dallas TX",
    "issue": "Water heater not heating",
    "trade": "Plumbing",
    "summary": "Customer reports cold water only, water heater may have failed.",
})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_mock_groq(response: str = VALID_EXTRACTION) -> MagicMock:
    mock = MagicMock(spec=GroqClient)
    mock.complete.return_value = response
    return mock


def make_mock_whisper(transcript: str = "Hi, my water heater stopped working.") -> MagicMock:
    mock = MagicMock()
    mock.transcribe.return_value = TranscriptionResult(
        transcript=transcript,
        duration_seconds=5.0,
    )
    return mock


@pytest.fixture
def shared_repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def mock_groq() -> MagicMock:
    return make_mock_groq()


@pytest.fixture
async def client(
    shared_repo: InMemoryJobRepository,
    mock_groq: MagicMock,
    tmp_path: Path,
):
    """
    Full test client with all three services sharing the same repository.
    Whisper and Groq are both mocked — no models, no API calls.
    """
    upload_svc = UploadService(repository=shared_repo, upload_dir=str(tmp_path))
    transcription_svc = TranscriptionService(
        repository=shared_repo,
        whisper_service=make_mock_whisper(),
    )
    extraction_svc = ExtractionService(
        repository=shared_repo,
        groq_client=mock_groq,
    )

    app.dependency_overrides[get_upload_service] = lambda: upload_svc
    app.dependency_overrides[get_transcription_service] = lambda: transcription_svc
    app.dependency_overrides[get_extraction_service] = lambda: extraction_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def upload_and_transcribe(client: AsyncClient, tmp_path: Path) -> str:
    """
    Helper: upload a WAV file, transcribe it, and return the job_id.
    After this helper the job is in TRANSCRIBED status, ready for extraction.
    """
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 1020)

    upload_resp = await client.post(
        "/api/v1/jobs/upload-audio",
        files={"file": ("call.wav", io.BytesIO(audio_file.read_bytes()), "audio/wav")},
    )
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.json()}"
    job_id = upload_resp.json()["job_id"]

    transcribe_resp = await client.post(f"/api/v1/jobs/{job_id}/transcribe")
    assert transcribe_resp.status_code == 200, f"Transcribe failed: {transcribe_resp.json()}"

    return job_id


# ---------------------------------------------------------------------------
# Happy path — HTTP status and response shape
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_200(client: AsyncClient, tmp_path: Path):
    """POST /{job_id}/extract returns HTTP 200 on success."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_extract_response_status_is_extracted(client: AsyncClient, tmp_path: Path):
    """Response body shows status=EXTRACTED."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.json()["status"] == "EXTRACTED"


@pytest.mark.asyncio
async def test_extract_response_contains_nested_extraction(
    client: AsyncClient, tmp_path: Path
):
    """Response body has an 'extraction' key containing the structured data."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    body = response.json()

    assert "extraction" in body
    assert isinstance(body["extraction"], dict)


@pytest.mark.asyncio
async def test_extract_response_extraction_fields(
    client: AsyncClient, tmp_path: Path
):
    """The nested extraction object contains all five required fields."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    extraction = response.json()["extraction"]

    for field in ("customer_name", "address", "issue", "trade", "summary"):
        assert field in extraction, f"Missing field: {field}"


@pytest.mark.asyncio
async def test_extract_response_customer_name(client: AsyncClient, tmp_path: Path):
    """Extracted customer_name in response matches LLM output."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.json()["extraction"]["customer_name"] == "Maria Garcia"


@pytest.mark.asyncio
async def test_extract_response_trade(client: AsyncClient, tmp_path: Path):
    """Extracted trade in response matches LLM output."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.json()["extraction"]["trade"] == "Plumbing"


@pytest.mark.asyncio
async def test_extract_response_schema(client: AsyncClient, tmp_path: Path):
    """Response body contains all expected top-level fields."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    body = response.json()

    for field in ("job_id", "status", "extraction", "extracted_at", "message"):
        assert field in body, f"Missing top-level field: {field}"


@pytest.mark.asyncio
async def test_extract_response_job_id_matches(client: AsyncClient, tmp_path: Path):
    """Response job_id matches the job that was extracted."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_extract_response_success_message(client: AsyncClient, tmp_path: Path):
    """Response contains the success message."""
    job_id = await upload_and_transcribe(client, tmp_path)
    response = await client.post(f"/api/v1/jobs/{job_id}/extract")
    assert response.json()["message"] == "Extraction completed successfully."


# ---------------------------------------------------------------------------
# Error paths — HTTP status codes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_404_for_unknown_job(client: AsyncClient):
    """POST with an unknown job_id returns HTTP 404."""
    response = await client.post("/api/v1/jobs/nonexistent-id/extract")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_extract_returns_400_when_no_transcript(
    shared_repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """POST on a job that hasn't been transcribed returns HTTP 400."""
    from app.schemas.job import Job, JobStatus

    now = datetime.now(tz=timezone.utc)
    job = Job(
        job_id="untranscribed-job",
        filename="untranscribed-job.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=1024,
        file_path=str(tmp_path / "call.wav"),
        status=JobStatus.UPLOADED,
        uploaded_at=now,
        updated_at=now,
        transcript=None,
    )
    await shared_repo.save(job)

    extraction_svc = ExtractionService(
        repository=shared_repo,
        groq_client=make_mock_groq(),
    )
    app.dependency_overrides[get_extraction_service] = lambda: extraction_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.post("/api/v1/jobs/untranscribed-job/extract")

    app.dependency_overrides.clear()
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_extract_returns_500_on_groq_failure(
    shared_repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """POST returns HTTP 500 when GroqClient raises ExtractionError."""
    from app.schemas.job import Job, JobStatus

    now = datetime.now(tz=timezone.utc)
    audio_file = tmp_path / "call.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    job = Job(
        job_id="failing-extract-job",
        filename="failing-extract-job.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=1024,
        file_path=str(audio_file),
        status=JobStatus.TRANSCRIBED,
        uploaded_at=now,
        updated_at=now,
        transcript="The boiler is making a loud noise.",
        transcribed_at=now,
        duration_seconds=4.0,
    )
    await shared_repo.save(job)

    failing_groq = MagicMock(spec=GroqClient)
    failing_groq.complete.side_effect = ExtractionError("Groq is down")

    extraction_svc = ExtractionService(
        repository=shared_repo,
        groq_client=failing_groq,
    )
    app.dependency_overrides[get_extraction_service] = lambda: extraction_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.post("/api/v1/jobs/failing-extract-job/extract")

    app.dependency_overrides.clear()
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_extract_returns_500_on_invalid_json(
    shared_repo: InMemoryJobRepository,
    tmp_path: Path,
):
    """POST returns HTTP 500 when Groq returns malformed JSON."""
    from app.schemas.job import Job, JobStatus

    now = datetime.now(tz=timezone.utc)
    audio_file = tmp_path / "call2.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    job = Job(
        job_id="bad-json-job",
        filename="bad-json-job.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=1024,
        file_path=str(audio_file),
        status=JobStatus.TRANSCRIBED,
        uploaded_at=now,
        updated_at=now,
        transcript="Hello I need help.",
        transcribed_at=now,
        duration_seconds=2.0,
    )
    await shared_repo.save(job)

    bad_groq = make_mock_groq(response="THIS IS NOT JSON {{{{")
    extraction_svc = ExtractionService(
        repository=shared_repo,
        groq_client=bad_groq,
    )
    app.dependency_overrides[get_extraction_service] = lambda: extraction_svc

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        response = await ac.post("/api/v1/jobs/bad-json-job/extract")

    app.dependency_overrides.clear()
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_extract_404_contains_detail(client: AsyncClient):
    """404 response includes a detail message."""
    response = await client.post("/api/v1/jobs/ghost/extract")
    body = response.json()
    assert "detail" in body
    assert len(body["detail"]) > 0
