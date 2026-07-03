"""
app/schemas/job.py

Pydantic schemas for Job — the central domain object of DispatchOps AI.

What is a schema vs a DB model?
- Schemas (this file) are Pydantic models. They define what data looks like
  when it moves between layers: API request → service → response.
- DB models (app/db/models/job.py, added in Feature 5) are SQLAlchemy models.
  They define what a row looks like in PostgreSQL.
- Keeping them separate means changing the database schema never forces a
  change to the API contract, and vice versa.

Why JobStatus as an Enum?
- A plain string like "uploaded" can be misspelled anywhere. An enum
  means the type checker catches it at write time, not at 2am in production.
- It documents every valid state in one place.
- Langfuse and logs always get a consistent, uppercase string.

Interview talking point:
"Job status is an enum with eight states that map the full pipeline lifecycle.
The route only ever sets UPLOADED. Each subsequent pipeline stage advances
the status one step forward. If any step fails, the status becomes FAILED or
NEEDS_HUMAN_REVIEW depending on whether human intervention is required."
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    """
    Lifecycle states for a DispatchOps AI job.

    Each state corresponds to one stage of the pipeline:

        UPLOADED          — Audio file received and saved. Pipeline not started.
        TRANSCRIBING      — Whisper is processing the audio file.
        TRANSCRIBED       — Whisper completed. Transcript stored on the job.
        EXTRACTING        — Groq LLM is extracting structured job details.
        RANKING           — Technician scoring engine is running.
        DISPATCHED        — Best technician selected, dispatch record created.
        COMPLETED         — Job fully resolved and closed.
        FAILED            — A pipeline stage failed and cannot be retried.
        NEEDS_HUMAN_REVIEW — LLM confidence was too low for auto-dispatch.
    """

    UPLOADED = "UPLOADED"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSCRIBED = "TRANSCRIBED"
    EXTRACTING = "EXTRACTING"
    RANKING = "RANKING"
    DISPATCHED = "DISPATCHED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"


class Job(BaseModel):
    """
    The core domain object representing one customer call / service job.

    This schema is used:
    - As the internal data transfer object between service and repository.
    - As the basis for the API response (various response schemas project from this).
    - Later, as the shape that maps to the PostgreSQL ORM model.

    Fields:
        job_id:            UUID string — unique identifier for this job.
        filename:          The stored filename on disk (UUID-based, e.g. "abc123.wav").
        original_filename: The original name the client sent (e.g. "customer_call.wav").
        content_type:      MIME type of the uploaded file (e.g. "audio/wav").
        file_size:         Size of the saved file in bytes.
        file_path:         Full relative path on disk (e.g. "uploads/2026/07/03/abc.wav").
        status:            Current pipeline stage.
        uploaded_at:       UTC timestamp when the file was received.
        updated_at:        UTC timestamp of the last status change.

        --- Added in Feature 3 ---
        transcript:        The full text transcribed by Whisper. None until transcribed.
        transcribed_at:    UTC timestamp when Whisper completed. None until transcribed.
        duration_seconds:  Duration of the audio clip in seconds. None until transcribed.
    """

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    filename: str = Field(..., description="Stored filename on disk")
    original_filename: str = Field(..., description="Original filename from the client")
    content_type: str = Field(..., description="MIME type of the audio file")
    file_size: int = Field(..., description="File size in bytes", ge=0)
    file_path: str = Field(..., description="Relative path to the saved audio file")
    status: JobStatus = Field(default=JobStatus.UPLOADED)
    uploaded_at: datetime = Field(..., description="UTC timestamp of upload")
    updated_at: datetime = Field(..., description="UTC timestamp of last update")

    # --- Feature 3: Transcription fields ---
    # All Optional — they are None until Whisper processes the audio.
    transcript: Optional[str] = Field(
        default=None,
        description="Full text transcript from Whisper. Populated after transcription.",
    )
    transcribed_at: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp when transcription completed.",
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of the audio file in seconds.",
    )

    model_config = {"from_attributes": True}  # Allows population from ORM objects in Feature 5


class JobResponse(BaseModel):
    """
    API response schema for a job upload.

    This is a deliberate subset of Job — we never expose file_path
    to the client. Internal storage paths are an implementation detail.

    Returned by POST /api/v1/jobs/upload-audio.
    """

    job_id: str = Field(..., description="Unique job identifier — use this to poll status")
    status: JobStatus = Field(..., description="Current pipeline status")
    filename: str = Field(..., description="Stored filename")
    original_filename: str = Field(..., description="Original filename from upload")
    content_type: str = Field(..., description="MIME type")
    file_size: int = Field(..., description="File size in bytes")
    uploaded_at: datetime = Field(..., description="UTC timestamp of upload")
    message: str = Field(..., description="Human-readable result message")


class TranscriptionResponse(BaseModel):
    """
    API response schema for a completed transcription.

    Returned by POST /api/v1/jobs/{job_id}/transcribe.

    We expose transcript and duration here because that is exactly what
    the caller requested — unlike JobResponse which is a general status view.
    file_path is still never exposed.
    """

    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Pipeline status — TRANSCRIBED on success")
    transcript: str = Field(..., description="Full text transcript of the audio")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    transcribed_at: datetime = Field(..., description="UTC timestamp of transcription")
    message: str = Field(..., description="Human-readable result message")
