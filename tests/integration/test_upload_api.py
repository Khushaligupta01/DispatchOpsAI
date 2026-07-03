"""
tests/integration/test_upload_api.py

Integration tests for POST /api/v1/jobs/upload-audio.

These tests exercise the full HTTP → service → repository chain
using FastAPI's async test client (via httpx + ASGI transport).

A key difference from unit tests:
- Unit tests call UploadService directly with bytes.
- Integration tests send a real multipart HTTP request and verify
  the HTTP response code, response body shape, and headers.

We override the get_upload_service dependency to inject a service
backed by a temporary directory and a fresh repository — so these
tests never write to the real uploads/ folder and never share state.
"""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient

from app.api.v1.dependencies import get_upload_service
from app.main import app
from app.repositories.job_repository import InMemoryJobRepository
from app.services.upload_service import UploadService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_upload_service(tmp_path):
    """
    UploadService with a fresh in-memory repo and isolated temp directory.

    Injected via app.dependency_overrides to replace the module-level
    singleton for the duration of each test.
    """
    return UploadService(
        repository=InMemoryJobRepository(),
        upload_dir=str(tmp_path),
    )


@pytest.fixture
async def client(isolated_upload_service: UploadService):
    """
    Async test client with the upload service dependency overridden.

    app.dependency_overrides replaces get_upload_service() with a lambda
    that returns our test service. After the test, the override is removed.
    """
    app.dependency_overrides[get_upload_service] = lambda: isolated_upload_service

    from httpx import ASGITransport
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

    # Clean up — remove the override so other tests are unaffected
    app.dependency_overrides.clear()


def make_upload_file(
    content: bytes = b"fake audio data",
    filename: str = "test_call.wav",
    content_type: str = "audio/wav",
):
    """Helper to build the files dict for httpx multipart upload."""
    return {"file": (filename, io.BytesIO(content), content_type)}


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_wav_returns_201(client: AsyncClient):
    """Valid WAV upload returns HTTP 201 Created."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(),
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_upload_response_contains_job_id(client: AsyncClient):
    """Response body contains a non-empty job_id."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(),
    )
    body = response.json()
    assert "job_id" in body
    assert len(body["job_id"]) > 0


@pytest.mark.asyncio
async def test_upload_response_status_is_uploaded(client: AsyncClient):
    """Response body shows status=UPLOADED."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(),
    )
    assert response.json()["status"] == "UPLOADED"


@pytest.mark.asyncio
async def test_upload_response_contains_message(client: AsyncClient):
    """Response body contains the success message."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(),
    )
    assert response.json()["message"] == "Audio uploaded successfully."


@pytest.mark.asyncio
async def test_upload_response_schema(client: AsyncClient):
    """Response body contains all expected fields."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="customer_call.wav"),
    )
    body = response.json()
    expected_fields = {
        "job_id", "status", "filename", "original_filename",
        "content_type", "file_size", "uploaded_at", "message",
    }
    assert expected_fields.issubset(body.keys())


@pytest.mark.asyncio
async def test_upload_original_filename_preserved(client: AsyncClient):
    """original_filename in response matches what was uploaded."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="my_hvac_call.wav"),
    )
    assert response.json()["original_filename"] == "my_hvac_call.wav"


@pytest.mark.asyncio
async def test_upload_mp3_returns_201(client: AsyncClient):
    """audio/mpeg upload returns HTTP 201."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="call.mp3", content_type="audio/mpeg"),
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_upload_m4a_returns_201(client: AsyncClient):
    """audio/x-m4a upload returns HTTP 201."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="call.m4a", content_type="audio/x-m4a"),
    )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_two_uploads_produce_different_job_ids(client: AsyncClient):
    """Uploading twice yields two distinct job_ids."""
    r1 = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="call1.wav"),
    )
    r2 = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="call2.wav"),
    )
    assert r1.json()["job_id"] != r2.json()["job_id"]


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upload_rejects_video_file_with_400(client: AsyncClient):
    """Unsupported content type returns HTTP 400."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="video.mp4", content_type="video/mp4"),
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_pdf_with_400(client: AsyncClient):
    """PDF upload returns HTTP 400."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(
            content=b"%PDF-1.4 fake",
            filename="doc.pdf",
            content_type="application/pdf",
        ),
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_rejects_oversized_file_with_413(client: AsyncClient):
    """File over 20 MB returns HTTP 413."""
    oversized = b"\x00" * (20 * 1024 * 1024 + 1)
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(content=oversized, filename="huge.wav"),
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_400_response_contains_detail(client: AsyncClient):
    """400 error response includes a detail field explaining the rejection."""
    response = await client.post(
        "/api/v1/jobs/upload-audio",
        files=make_upload_file(filename="video.mp4", content_type="video/mp4"),
    )
    body = response.json()
    assert "detail" in body
    assert len(body["detail"]) > 0
