"""
tests/unit/test_upload_service.py

Unit tests for UploadService.

These tests verify the service's business logic in complete isolation:
- No HTTP stack (no TestClient, no FastAPI)
- No real filesystem (we test with a tmp_path fixture)
- No real repository (we inject InMemoryJobRepository directly)

Why test at the service layer?
- Service tests run faster than integration tests (no HTTP overhead).
- Service tests are more precise — a failure pinpoints the exact logic,
  not somewhere in the HTTP → service → storage chain.
- If the route layer changes (e.g., REST → GraphQL), these tests still work.
"""

from __future__ import annotations

import pytest

from app.repositories.job_repository import InMemoryJobRepository
from app.schemas.job import JobStatus
from app.services.upload_service import (
    FileTooLargeError,
    InvalidFileTypeError,
    UploadService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo() -> InMemoryJobRepository:
    """A fresh in-memory repository for each test."""
    return InMemoryJobRepository()


@pytest.fixture
def service(repo: InMemoryJobRepository, tmp_path) -> UploadService:
    """
    UploadService backed by an in-memory repo and a temporary directory.

    tmp_path is a pytest built-in fixture that provides a unique temp
    directory per test — no cleanup needed, no leftover files.
    """
    return UploadService(repository=repo, upload_dir=str(tmp_path))


def make_audio_bytes(size: int = 1024) -> bytes:
    """Generate dummy audio bytes for testing."""
    return b"RIFF" + b"\x00" * (size - 4)  # Minimal WAV-like header


# ---------------------------------------------------------------------------
# Valid upload tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_returns_job(service: UploadService):
    """A valid WAV upload returns a Job with status UPLOADED."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="test_call.wav",
        content_type="audio/wav",
        request_id="req-001",
    )

    assert job.job_id is not None
    assert job.status == JobStatus.UPLOADED
    assert job.original_filename == "test_call.wav"
    assert job.content_type == "audio/wav"
    assert job.file_size == 1024


@pytest.mark.asyncio
async def test_upload_generates_unique_job_ids(service: UploadService):
    """Two uploads produce two different job_ids."""
    job1 = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call1.wav",
        content_type="audio/wav",
        request_id="req-001",
    )
    job2 = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call2.wav",
        content_type="audio/wav",
        request_id="req-002",
    )
    assert job1.job_id != job2.job_id


@pytest.mark.asyncio
async def test_upload_saves_file_to_disk(service: UploadService, tmp_path):
    """The uploaded file is actually written to the date-based directory."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(2048),
        original_filename="test_call.wav",
        content_type="audio/wav",
        request_id="req-001",
    )

    saved_path = tmp_path / job.file_path.lstrip("/").replace(str(tmp_path), "").lstrip("\\").lstrip("/")
    # Verify a file exists at the reported path
    from pathlib import Path
    assert Path(job.file_path).exists()


@pytest.mark.asyncio
async def test_upload_persists_to_repository(
    service: UploadService, repo: InMemoryJobRepository
):
    """After upload, the job can be retrieved from the repository."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call.wav",
        content_type="audio/wav",
        request_id="req-001",
    )

    stored = await repo.get_by_id(job.job_id)
    assert stored is not None
    assert stored.job_id == job.job_id


@pytest.mark.asyncio
async def test_upload_mp3_content_type(service: UploadService):
    """audio/mpeg is a valid content type."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call.mp3",
        content_type="audio/mpeg",
        request_id="req-001",
    )
    assert job.status == JobStatus.UPLOADED


@pytest.mark.asyncio
async def test_upload_m4a_content_type(service: UploadService):
    """audio/x-m4a is a valid content type."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call.m4a",
        content_type="audio/x-m4a",
        request_id="req-001",
    )
    assert job.status == JobStatus.UPLOADED


@pytest.mark.asyncio
async def test_filename_uses_uuid_not_original(service: UploadService):
    """Stored filename is a UUID, not the original filename."""
    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="../../etc/passwd.wav",  # Path traversal attempt
        content_type="audio/wav",
        request_id="req-001",
    )
    # The stored filename must NOT contain the original name
    assert "passwd" not in job.filename
    assert "etc" not in job.filename
    # It should end with .wav
    assert job.filename.endswith(".wav")


@pytest.mark.asyncio
async def test_file_path_is_date_organized(service: UploadService):
    """File path includes YYYY/MM/DD date directory structure."""
    from datetime import datetime, timezone
    today = datetime.now(tz=timezone.utc)

    job = await service.upload_audio(
        file_data=make_audio_bytes(),
        original_filename="call.wav",
        content_type="audio/wav",
        request_id="req-001",
    )

    assert str(today.year) in job.file_path
    assert f"{today.month:02d}" in job.file_path
    assert f"{today.day:02d}" in job.file_path


# ---------------------------------------------------------------------------
# Validation failure tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rejects_unsupported_content_type(service: UploadService):
    """video/mp4 is rejected with InvalidFileTypeError."""
    with pytest.raises(InvalidFileTypeError) as exc_info:
        await service.upload_audio(
            file_data=make_audio_bytes(),
            original_filename="video.mp4",
            content_type="video/mp4",
            request_id="req-001",
        )
    assert "video/mp4" in exc_info.value.message


@pytest.mark.asyncio
async def test_rejects_pdf_upload(service: UploadService):
    """application/pdf is rejected with InvalidFileTypeError."""
    with pytest.raises(InvalidFileTypeError):
        await service.upload_audio(
            file_data=b"%PDF-1.4 fake content",
            original_filename="document.pdf",
            content_type="application/pdf",
            request_id="req-001",
        )


@pytest.mark.asyncio
async def test_rejects_oversized_file(service: UploadService):
    """A file over 20 MB is rejected with FileTooLargeError."""
    oversized = b"\x00" * (20 * 1024 * 1024 + 1)  # 20 MB + 1 byte
    with pytest.raises(FileTooLargeError) as exc_info:
        await service.upload_audio(
            file_data=oversized,
            original_filename="huge.wav",
            content_type="audio/wav",
            request_id="req-001",
        )
    assert "20" in exc_info.value.message  # error mentions the limit


@pytest.mark.asyncio
async def test_accepts_exactly_20mb_file(service: UploadService):
    """A file of exactly 20 MB is accepted (boundary condition)."""
    exactly_20mb = b"\x00" * (20 * 1024 * 1024)
    job = await service.upload_audio(
        file_data=exactly_20mb,
        original_filename="max_size.wav",
        content_type="audio/wav",
        request_id="req-001",
    )
    assert job.file_size == 20 * 1024 * 1024
