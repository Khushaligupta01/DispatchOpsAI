"""
app/services/extraction_service.py

Extraction service — orchestrates LLM-based structured data extraction.

PIPELINE STEPS:
  1. Fetch the Job from the repository.
  2. Verify the job exists and has a transcript.
  3. Mark the Job as EXTRACTING.
  4. Format the prompt with the transcript.
  5. Call GroqClient → raw JSON string.
  6. Parse and validate JSON into JobExtraction (Pydantic).
  7. Store JobExtraction as job.extraction (nested object).
  8. Mark the Job as EXTRACTED.
  9. Persist and return the Job.

WHY NESTED EXTRACTION (job.extraction) NOT FLAT FIELDS?
---------------------------------------------------------
Storing the extraction as a nested JobExtraction object on the Job keeps
the domain model clean. job.extraction.trade is unambiguous — it came
from LLM extraction. A flat job.trade field has no clear provenance.

If a future pipeline stage also extracts data (e.g. a second LLM pass),
we add job.refined_extraction without touching existing fields.

WHY DOES THE SERVICE NOT PARSE JSON DIRECTLY?
---------------------------------------------
All JSON parsing goes through JobExtraction (Pydantic). This gives us:
- Required field enforcement — missing key → ValidationError → ExtractionError.
- Type validation — every field must be a string.
- A clean typed object for downstream use; no dict["key"] access.

Interview talking point:
"The extraction service follows the TRANSCRIBING/TRANSCRIBED pattern:
mark in-progress before the LLM call, mark complete after. If parsing
fails, the job is marked FAILED and ExtractionError propagates to the route.
The repository always reflects what actually happened."
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pydantic import ValidationError

from app.llm.groq_client import GroqClient
from app.llm.prompts import EXTRACTION_PROMPT, EXTRACTION_PROMPT_VERSION
from app.repositories.job_repository import AbstractJobRepository
from app.schemas.job import Job, JobExtraction, JobStatus
from app.utils.exceptions import DispatchOpsError, ExtractionError, JobNotFoundError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptMissingError(DispatchOpsError):
    """Raised when extract_job is called on a job that has no transcript yet."""
    pass


class ExtractionService:
    """
    Orchestrates LLM extraction for a single job.

    Args:
        repository:  Persists Job state.
        groq_client: Sends prompts to the Groq API.
    """

    def __init__(
        self,
        repository: AbstractJobRepository,
        groq_client: GroqClient,
    ) -> None:
        self._repo = repository
        self._groq = groq_client

    async def extract_job(self, job_id: str) -> Job:
        """
        Run the LLM extraction pipeline for the given job.

        Args:
            job_id: UUID string of the job to extract.

        Returns:
            Updated Job with status=EXTRACTED and extraction populated.

        Raises:
            JobNotFoundError:       No job with this job_id exists.
            TranscriptMissingError: Job has no transcript to extract from.
            ExtractionError:        LLM call failed or returned invalid JSON.
        """
        logger.info("Extraction pipeline started", extra={"job_id": job_id})

        # Step 1 — Fetch
        job = await self._repo.get_by_id(job_id)
        if job is None:
            logger.warning("Extraction failed — job not found", extra={"job_id": job_id})
            raise JobNotFoundError(
                message=f"Job '{job_id}' not found.",
                detail="Ensure the job was created and transcribed first.",
            )

        # Step 2 — Verify transcript
        if not job.transcript:
            logger.warning(
                "Extraction failed — transcript missing",
                extra={"job_id": job_id, "status": job.status},
            )
            raise TranscriptMissingError(
                message=f"Job '{job_id}' has no transcript.",
                detail=(
                    f"Current status: {job.status}. "
                    "Call POST /api/v1/jobs/{job_id}/transcribe first."
                ),
            )

        # Step 3 — Mark EXTRACTING
        await self._repo.update_status(job_id, JobStatus.EXTRACTING)
        logger.info(
            "Extraction started",
            extra={
                "job_id": job_id,
                "transcript_length": len(job.transcript),
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            },
        )

        # Step 4 — Call LLM
        prompt = EXTRACTION_PROMPT.replace("<<TRANSCRIPT>>", job.transcript)
        try:
            raw_response = self._groq.complete(prompt)
        except ExtractionError:
            await self._repo.update_status(job_id, JobStatus.FAILED)
            logger.error("Extraction failed — LLM error", extra={"job_id": job_id})
            raise

        # Step 5 — Parse and validate response
        extraction = self._parse_llm_response(job_id, raw_response)

        # Step 6 — Build updated Job with nested extraction
        now = datetime.now(tz=timezone.utc)
        updated_job = job.model_copy(
            update={
                "extraction": extraction,
                "extracted_at": now,
                "updated_at": now,
                "status": JobStatus.EXTRACTED,
            }
        )

        # Step 7 — Persist
        persisted_job = await self._repo.update_job(updated_job)
        logger.info(
            "Extraction completed",
            extra={
                "job_id": job_id,
                "trade": extraction.trade,
                "prompt_version": EXTRACTION_PROMPT_VERSION,
            },
        )
        return persisted_job or updated_job

    def _parse_llm_response(self, job_id: str, raw: str) -> JobExtraction:
        """
        Parse the raw LLM response string into a validated JobExtraction.

        Strips markdown code fences defensively — some models wrap JSON in
        ```json ... ``` blocks despite instructions not to.

        Raises:
            ExtractionError: JSON is malformed or Pydantic validation fails.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error(
                "LLM returned invalid JSON",
                extra={
                    "job_id": job_id,
                    "raw_response": raw[:500],
                    "error": str(exc),
                },
            )
            raise ExtractionError(
                message="LLM returned invalid JSON.",
                detail=f"Parse error: {exc}. Raw (truncated): {raw[:200]}",
            ) from exc

        try:
            return JobExtraction(**data)
        except ValidationError as exc:
            logger.error(
                "LLM JSON failed schema validation",
                extra={"job_id": job_id, "validation_errors": str(exc)},
            )
            raise ExtractionError(
                message="LLM response failed schema validation.",
                detail=str(exc),
            ) from exc
