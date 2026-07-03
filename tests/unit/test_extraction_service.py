"""
tests/unit/test_extraction_service.py

Unit tests for ExtractionService.

Tests the orchestration logic in complete isolation:
- No HTTP layer
- No real Groq API (GroqClient is mocked)
- InMemoryJobRepository used directly (no DB)

What we test:
- Successful extraction populates job.extraction with a JobExtraction object.
- Successful extraction sets status to EXTRACTED.
- Successful extraction sets extracted_at timestamp.
- Successful extraction persists to the repository.
- JobNotFoundError raised for unknown job_id.
- TranscriptMissingError raised when job has no transcript.
- ExtractionError propagates when GroqClient fails, job marked FAILED.
- ExtractionError raised for invalid JSON from LLM.
- ExtractionError raised for valid JSON but missing required fields.
- Markdown code fences are stripped from LLM response before parsing.
- Status transitions: TRANSCRIBED → EXTRACTING → EXTRACTED.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.llm.groq_client import GroqClient
from app.repositories.job_repository import InMemoryJobRepository
from app.schemas.job import Job, JobExtraction, JobStatus
from app.services.extraction_service import ExtractionService, TranscriptMissingError
from app.utils.exceptions import ExtractionError, JobNotFoundError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_JSON = json.dumps({
    "customer_name": "Sarah Johnson",
    "address": "142 Oak Street, Austin TX",
    "issue": "HVAC unit not cooling",
    "trade": "HVAC",
    "summary": "Customer reports AC stopped working yesterday evening.",
})


def make_job(
    job_id: str = "job-extract-001",
    transcript: str = "Hi, my name is Sarah. My AC stopped working.",
    status: JobStatus = JobStatus.TRANSCRIBED,
) -> Job:
    """Create a Job ready for extraction (has a transcript)."""
    now = datetime.now(tz=timezone.utc)
    return Job(
        job_id=job_id,
        filename=f"{job_id}.wav",
        original_filename="call.wav",
        content_type="audio/wav",
        file_size=2048,
        file_path=f"uploads/2026/07/01/{job_id}.wav",
        status=status,
        uploaded_at=now,
        updated_at=now,
        transcript=transcript,
        transcribed_at=now,
        duration_seconds=8.5,
    )


def make_mock_groq(response: str = VALID_JSON) -> MagicMock:
    """Build a GroqClient mock returning a fixed response string."""
    mock = MagicMock(spec=GroqClient)
    mock.complete.return_value = response
    return mock


@pytest.fixture
def repo() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def mock_groq() -> MagicMock:
    return make_mock_groq()


@pytest.fixture
def service(repo: InMemoryJobRepository, mock_groq: MagicMock) -> ExtractionService:
    return ExtractionService(repository=repo, groq_client=mock_groq)


# ---------------------------------------------------------------------------
# Happy path — extraction result
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_job_with_nested_extraction(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Successful extraction sets job.extraction to a JobExtraction instance."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extraction is not None
    assert isinstance(job.extraction, JobExtraction)


@pytest.mark.asyncio
async def test_extract_customer_name(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Extracted customer_name matches the LLM response."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extraction.customer_name == "Sarah Johnson"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_extract_address(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Extracted address matches the LLM response."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extraction.address == "142 Oak Street, Austin TX"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_extract_issue(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Extracted issue matches the LLM response."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extraction.issue == "HVAC unit not cooling"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_extract_trade(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Extracted trade matches the LLM response."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extraction.trade == "HVAC"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_extract_summary(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Extracted summary matches the LLM response."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert "AC stopped" in job.extraction.summary  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Happy path — job state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_sets_status_extracted(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Successful extraction sets job status to EXTRACTED."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.status == JobStatus.EXTRACTED


@pytest.mark.asyncio
async def test_extract_sets_extracted_at(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """Successful extraction populates extracted_at timestamp."""
    await repo.save(make_job())
    job = await service.extract_job("job-extract-001")

    assert job.extracted_at is not None
    assert isinstance(job.extracted_at, datetime)


@pytest.mark.asyncio
async def test_extract_persists_to_repository(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """After extraction, the updated job is retrievable from the repository."""
    await repo.save(make_job())
    await service.extract_job("job-extract-001")

    stored = await repo.get_by_id("job-extract-001")
    assert stored is not None
    assert stored.status == JobStatus.EXTRACTED
    assert stored.extraction is not None
    assert stored.extraction.trade == "HVAC"


@pytest.mark.asyncio
async def test_extract_groq_called_with_transcript(
    service: ExtractionService, repo: InMemoryJobRepository, mock_groq: MagicMock
):
    """GroqClient.complete() is called with a prompt containing the transcript."""
    transcript = "Hello, I need a plumber. Pipe burst in kitchen."
    await repo.save(make_job(transcript=transcript))
    await service.extract_job("job-extract-001")

    call_args = mock_groq.complete.call_args[0][0]
    assert transcript in call_args


@pytest.mark.asyncio
async def test_extract_status_transitions_through_extracting(
    service: ExtractionService, repo: InMemoryJobRepository, mock_groq: MagicMock
):
    """EXTRACTING status is set before the LLM call, EXTRACTED after."""
    await repo.save(make_job())
    observed: list = []

    def capture_and_respond(prompt: str) -> str:
        job = repo._store.get("job-extract-001")
        if job:
            observed.append(job.status)
        return VALID_JSON

    mock_groq.complete.side_effect = capture_and_respond
    job = await service.extract_job("job-extract-001")

    assert JobStatus.EXTRACTING in observed
    assert job.status == JobStatus.EXTRACTED


# ---------------------------------------------------------------------------
# JSON parsing — code fence stripping
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_strips_markdown_code_fences(
    repo: InMemoryJobRepository
):
    """JSON wrapped in ```json ... ``` is parsed correctly."""
    fenced = f"```json\n{VALID_JSON}\n```"
    groq = make_mock_groq(response=fenced)
    svc = ExtractionService(repository=repo, groq_client=groq)

    await repo.save(make_job())
    job = await svc.extract_job("job-extract-001")

    assert job.extraction is not None
    assert job.extraction.trade == "HVAC"


@pytest.mark.asyncio
async def test_extract_strips_plain_code_fences(
    repo: InMemoryJobRepository
):
    """JSON wrapped in ``` ... ``` (no language tag) is parsed correctly."""
    fenced = f"```\n{VALID_JSON}\n```"
    groq = make_mock_groq(response=fenced)
    svc = ExtractionService(repository=repo, groq_client=groq)

    await repo.save(make_job())
    job = await svc.extract_job("job-extract-001")

    assert job.extraction is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_raises_job_not_found(service: ExtractionService):
    """extract_job raises JobNotFoundError for an unknown job_id."""
    with pytest.raises(JobNotFoundError) as exc_info:
        await service.extract_job("ghost-job")

    assert "ghost-job" in exc_info.value.message


@pytest.mark.asyncio
async def test_raises_transcript_missing_when_no_transcript(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """extract_job raises TranscriptMissingError when job.transcript is None."""
    await repo.save(make_job(transcript=None))  # type: ignore[arg-type]
    with pytest.raises(TranscriptMissingError) as exc_info:
        await service.extract_job("job-extract-001")

    assert "job-extract-001" in exc_info.value.message


@pytest.mark.asyncio
async def test_raises_transcript_missing_for_empty_transcript(
    service: ExtractionService, repo: InMemoryJobRepository
):
    """extract_job raises TranscriptMissingError when transcript is empty string."""
    await repo.save(make_job(transcript=""))
    with pytest.raises(TranscriptMissingError):
        await service.extract_job("job-extract-001")


@pytest.mark.asyncio
async def test_job_marked_failed_on_llm_error(
    repo: InMemoryJobRepository
):
    """When GroqClient raises ExtractionError, job status is set to FAILED."""
    failing_groq = MagicMock(spec=GroqClient)
    failing_groq.complete.side_effect = ExtractionError("LLM timeout")
    svc = ExtractionService(repository=repo, groq_client=failing_groq)

    await repo.save(make_job())
    with pytest.raises(ExtractionError):
        await svc.extract_job("job-extract-001")

    stored = await repo.get_by_id("job-extract-001")
    assert stored is not None
    assert stored.status == JobStatus.FAILED


@pytest.mark.asyncio
async def test_raises_extraction_error_for_invalid_json(
    repo: InMemoryJobRepository
):
    """ExtractionError raised when LLM returns malformed JSON."""
    bad_groq = make_mock_groq(response="not valid JSON at all {{")
    svc = ExtractionService(repository=repo, groq_client=bad_groq)

    await repo.save(make_job())
    with pytest.raises(ExtractionError) as exc_info:
        await svc.extract_job("job-extract-001")

    assert "invalid JSON" in exc_info.value.message


@pytest.mark.asyncio
async def test_raises_extraction_error_for_missing_required_field(
    repo: InMemoryJobRepository
):
    """ExtractionError raised when JSON is valid but missing a required field."""
    # Missing 'trade' field
    incomplete = json.dumps({
        "customer_name": "Bob",
        "address": "Unknown",
        "issue": "Leak",
        "summary": "Bob has a leak.",
        # "trade" intentionally omitted
    })
    bad_groq = make_mock_groq(response=incomplete)
    svc = ExtractionService(repository=repo, groq_client=bad_groq)

    await repo.save(make_job())
    with pytest.raises(ExtractionError) as exc_info:
        await svc.extract_job("job-extract-001")

    assert "schema validation" in exc_info.value.message


@pytest.mark.asyncio
async def test_extraction_error_propagates_from_groq(
    service: ExtractionService, repo: InMemoryJobRepository, mock_groq: MagicMock
):
    """ExtractionError from GroqClient propagates out of extract_job."""
    mock_groq.complete.side_effect = ExtractionError("API rate limit")
    await repo.save(make_job())

    with pytest.raises(ExtractionError):
        await service.extract_job("job-extract-001")
