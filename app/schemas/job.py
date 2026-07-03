"""
app/schemas/job.py

Pydantic schemas for Job — the central domain object of DispatchOps AI.

SCHEMA vs DB MODEL:
-------------------
- Schemas (this file): Pydantic models used as data transfer objects between
  API, service, and repository layers.
- DB models (app/db/models/job.py, Feature 5+): SQLAlchemy ORM definitions.

Keeping them separate means the database schema and the API contract can
evolve independently.

JOB DESIGN — WHY NESTED EXTRACTION?
-------------------------------------
The extraction result (JobExtraction) is stored as a nested object on Job,
not as flat fields. This keeps the domain model clean:

  job.extraction.customer_name   ← clear provenance
  job.customer_name              ← ambiguous (from where? when?)

A nested model also makes it trivial to add future AI stages (e.g.
job.ranking_result, job.dispatch_result) without expanding the flat field list.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    """
    Lifecycle states for a DispatchOps AI job.

    Pipeline order:
        UPLOADED → TRANSCRIBING → TRANSCRIBED → EXTRACTING → EXTRACTED
        → RANKING → DISPATCHED → COMPLETED

    Terminal states (no further transitions):
        FAILED, NEEDS_HUMAN_REVIEW
    """

    UPLOADED = "UPLOADED"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSCRIBED = "TRANSCRIBED"
    EXTRACTING = "EXTRACTING"
    EXTRACTED = "EXTRACTED"
    RANKING = "RANKING"
    DISPATCHED = "DISPATCHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class JobExtraction(BaseModel):
    """
    Structured dispatch information extracted from a call transcript by the LLM.

    This schema serves two purposes:
    1. Validates the raw JSON string returned by the LLM — if the LLM returns
       malformed JSON or omits a required field, Pydantic raises a ValidationError
       that the service catches and converts to ExtractionError.
    2. Provides a typed, self-documenting shape for downstream layers
       (ranking, dispatch) that need the extracted job details.

    WHY ALL FIELDS REQUIRED (no Optional)?
    ----------------------------------------
    The extraction prompt instructs the LLM to return "Unknown" instead of
    null or omitting a key. This means Pydantic always receives a string for
    every field — never None. Downstream code can safely do `extraction.trade`
    without an Optional check.

    Interview talking point:
    "The LLM returns raw JSON. We parse it into JobExtraction using Pydantic.
    Malformed JSON raises JSONDecodeError; a missing required field raises
    ValidationError. Both are caught and converted to ExtractionError. We
    never pass bad data downstream — it fails loudly at the boundary."
    """

    customer_name: str = Field(
        ..., description="Full name of the customer, or 'Unknown'."
    )
    address: str = Field(
        ..., description="Service address, or 'Unknown'."
    )
    issue: str = Field(
        ..., description="Specific problem described by the customer."
    )
    trade: str = Field(
        ...,
        description="Trade type: HVAC, Plumbing, Electrical, Roofing, General, or Unknown.",
    )
    summary: str = Field(
        ..., description="One-sentence summary of the job."
    )


class Job(BaseModel):
    """
    The core domain object representing one customer call / service job.

    Flows through: API → Service → Repository → API response.
    Maps to the PostgreSQL jobs table in Feature 5.

    FIELDS BY PIPELINE STAGE:
    --------------------------
    Created at upload (Feature 2):
        job_id, filename, original_filename, content_type, file_size,
        file_path, status, uploaded_at, updated_at

    Populated after transcription (Feature 3):
        transcript, transcribed_at, duration_seconds

    Populated after extraction (Feature 4):
        extraction, extracted_at
    """

    # --- Core (Feature 2) ---
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    filename: str = Field(..., description="UUID-based stored filename on disk")
    original_filename: str = Field(..., description="Original filename from the client")
    content_type: str = Field(..., description="MIME type of the audio file")
    file_size: int = Field(..., description="File size in bytes", ge=0)
    file_path: str = Field(..., description="Relative path to the saved audio file")
    status: JobStatus = Field(default=JobStatus.UPLOADED)
    uploaded_at: datetime = Field(..., description="UTC timestamp of upload")
    updated_at: datetime = Field(..., description="UTC timestamp of last update")

    # --- Transcription (Feature 3) ---
    transcript: Optional[str] = Field(
        default=None,
        description="Full text transcript from Whisper. None until transcribed.",
    )
    transcribed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when transcription completed.",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Audio duration in seconds. None until transcribed.",
    )

    # --- Extraction (Feature 4) ---
    extraction: Optional[JobExtraction] = Field(
        default=None,
        description=(
            "Structured dispatch data extracted by the LLM. "
            "None until extraction completes."
        ),
    )
    extracted_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when extraction completed.",
    )

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# API Response Schemas
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """
    Upload response — POST /api/v1/jobs/upload-audio.

    Subset of Job. file_path is intentionally excluded — internal storage
    paths are never exposed to clients.
    """

    job_id: str = Field(..., description="Use this to poll status or advance the pipeline")
    status: JobStatus
    filename: str
    original_filename: str
    content_type: str
    file_size: int
    uploaded_at: datetime
    message: str


class TranscriptionResponse(BaseModel):
    """
    Transcription response — POST /api/v1/jobs/{job_id}/transcribe.
    """

    job_id: str
    status: JobStatus
    transcript: str
    duration_seconds: float
    transcribed_at: datetime
    message: str


class ExtractionResponse(BaseModel):
    """
    Extraction response — POST /api/v1/jobs/{job_id}/extract.

    Nests the JobExtraction object under the 'extraction' key to mirror
    the domain model and make the response structure self-documenting.

    Response shape:
    {
        "job_id": "...",
        "status": "EXTRACTED",
        "extraction": {
            "customer_name": "...",
            "address": "...",
            "issue": "...",
            "trade": "...",
            "summary": "..."
        },
        "extracted_at": "...",
        "message": "..."
    }
    """

    job_id: str
    status: JobStatus
    extraction: JobExtraction
    extracted_at: datetime
    message: str
