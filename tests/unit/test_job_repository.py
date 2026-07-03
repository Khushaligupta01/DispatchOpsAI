"""
tests/unit/test_job_repository.py

Unit tests for InMemoryJobRepository.

Tests every method of the repository in isolation — no service, no HTTP.
This verifies the data access layer works correctly before connecting
it to anything else.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.repositories.job_repository import InMemoryJobRepository
from app.schemas.job import Job, JobStatus


def make_job(job_id: str = "test-job-1") -> Job:
    """Helper to create a minimal valid Job for testing."""
    now = datetime.now(tz=timezone.utc)
    return Job(
        job_id=job_id,
        filename=f"{job_id}.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=1024,
        file_path=f"uploads/2026/07/01/{job_id}.wav",
        status=JobStatus.UPLOADED,
        uploaded_at=now,
        updated_at=now,
    )


@pytest.fixture
def repo() -> InMemoryJobRepository:
    """Fresh repository instance for each test."""
    return InMemoryJobRepository()


# ---------------------------------------------------------------------------
# save()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_returns_job(repo: InMemoryJobRepository):
    """save() returns the saved job."""
    job = make_job()
    result = await repo.save(job)
    assert result.job_id == job.job_id


@pytest.mark.asyncio
async def test_save_persists_job(repo: InMemoryJobRepository):
    """After save(), the job can be retrieved."""
    job = make_job("job-abc")
    await repo.save(job)
    retrieved = await repo.get_by_id("job-abc")
    assert retrieved is not None
    assert retrieved.job_id == "job-abc"


# ---------------------------------------------------------------------------
# get_by_id()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing(repo: InMemoryJobRepository):
    """get_by_id() returns None when the job doesn't exist."""
    result = await repo.get_by_id("does-not-exist")
    assert result is None


@pytest.mark.asyncio
async def test_get_by_id_returns_correct_job(repo: InMemoryJobRepository):
    """get_by_id() returns the exact job that was saved."""
    job = make_job("job-xyz")
    await repo.save(job)

    result = await repo.get_by_id("job-xyz")
    assert result is not None
    assert result.original_filename == "call.wav"


# ---------------------------------------------------------------------------
# get_all()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_all_empty(repo: InMemoryJobRepository):
    """get_all() returns an empty list when no jobs exist."""
    result = await repo.get_all()
    assert result == []


@pytest.mark.asyncio
async def test_get_all_returns_all_jobs(repo: InMemoryJobRepository):
    """get_all() returns every saved job."""
    await repo.save(make_job("job-1"))
    await repo.save(make_job("job-2"))
    await repo.save(make_job("job-3"))

    all_jobs = await repo.get_all()
    assert len(all_jobs) == 3
    ids = {j.job_id for j in all_jobs}
    assert ids == {"job-1", "job-2", "job-3"}


# ---------------------------------------------------------------------------
# update_status()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_status_changes_status(repo: InMemoryJobRepository):
    """update_status() advances the job to the new status."""
    job = make_job("job-status-test")
    await repo.save(job)

    updated = await repo.update_status("job-status-test", JobStatus.TRANSCRIBING)
    assert updated is not None
    assert updated.status == JobStatus.TRANSCRIBING


@pytest.mark.asyncio
async def test_update_status_persists_change(repo: InMemoryJobRepository):
    """After update_status(), get_by_id() returns the updated status."""
    await repo.save(make_job("job-persist"))
    await repo.update_status("job-persist", JobStatus.EXTRACTING)

    retrieved = await repo.get_by_id("job-persist")
    assert retrieved is not None
    assert retrieved.status == JobStatus.EXTRACTING


@pytest.mark.asyncio
async def test_update_status_returns_none_for_missing(repo: InMemoryJobRepository):
    """update_status() returns None when the job doesn't exist."""
    result = await repo.update_status("ghost-job", JobStatus.FAILED)
    assert result is None


@pytest.mark.asyncio
async def test_update_status_does_not_affect_other_jobs(repo: InMemoryJobRepository):
    """Updating one job's status does not change other jobs."""
    await repo.save(make_job("job-a"))
    await repo.save(make_job("job-b"))

    await repo.update_status("job-a", JobStatus.DISPATCHED)

    job_b = await repo.get_by_id("job-b")
    assert job_b is not None
    assert job_b.status == JobStatus.UPLOADED  # Unchanged
