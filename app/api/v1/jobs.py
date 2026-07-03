"""
app/api/v1/jobs.py

Job-related API routes for DispatchOps AI.

This module contains only one endpoint in Feature 2:
    POST /api/v1/jobs/upload-audio

WHAT THIS ROUTE DOES (and does NOT do):
- Reads the uploaded file from the multipart request.           ✓
- Generates a request_id for log correlation.                  ✓
- Calls UploadService to validate, save, and create the Job.   ✓
- Returns a structured JSON response.                          ✓
- Contains zero business logic.                                ✓ (intentional)
- Touches the filesystem directly.                             ✗ (that's the service)
- Validates content type or file size.                         ✗ (that's the service)
- Knows about the repository.                                  ✗ (that's the service)

WHY INJECT UploadService VIA Depends()?
---------------------------------------
FastAPI resolves dependencies before calling the route handler.
In tests, we override get_upload_service() to inject a service backed
by a mock repository — no real disk writes, no real DB needed.
In production, FastAPI injects the real service automatically.
No import changes. No monkey-patching. Clean.

Interview talking point:
"The route handler is six lines of real logic. It reads the file,
generates a request ID, calls the service, and returns the result.
All validation, all file I/O, all error handling lives in UploadService.
The route is purely an HTTP adapter."
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas.job import JobResponse
from app.services.upload_service import (
    FileTooLargeError,
    InvalidFileTypeError,
    UploadService,
)
from app.utils.logger import get_logger

from .dependencies import get_upload_service

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post(
    "/upload-audio",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a customer call recording",
    description=(
        "Accepts an audio file (WAV, MP3, M4A) up to 20 MB. "
        "Validates the file, saves it to disk, and creates a new Job record. "
        "Returns a job_id that can be used to poll the pipeline status."
    ),
    responses={
        201: {"description": "Audio uploaded and job created successfully"},
        400: {"description": "Unsupported file type"},
        413: {"description": "File exceeds 20 MB limit"},
        500: {"description": "Unexpected server error"},
    },
)
async def upload_audio(
    file: UploadFile = File(
        ...,
        description="Audio file to upload. Supported formats: WAV, MP3, M4A.",
    ),
    upload_service: UploadService = Depends(get_upload_service),
) -> JobResponse:
    """
    Upload a customer call recording and create a new job.

    The file is validated, saved under uploads/YYYY/MM/DD/<uuid>.<ext>,
    and registered as a Job with status=UPLOADED.

    The returned job_id is used to track the job through the AI pipeline
    in subsequent features (transcription → extraction → ranking → dispatch).
    """
    # Generate a request ID for correlating all log lines from this request
    request_id = str(uuid.uuid4())

    logger.info(
        "Upload request received",
        extra={
            "request_id": request_id,
            "audio_filename": file.filename,
            "content_type": file.content_type,
        },
    )

    # Read file bytes — done here because UploadFile is an HTTP concern.
    # The service receives plain bytes and knows nothing about HTTP.
    try:
        file_data = await file.read()
    except Exception as exc:
        logger.error(
            "Failed to read uploaded file",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read uploaded file.",
        ) from exc

    # Delegate all business logic to the service
    try:
        job = await upload_service.upload_audio(
            file_data=file_data,
            original_filename=file.filename or "unknown.wav",
            content_type=file.content_type or "application/octet-stream",
            request_id=request_id,
        )
    except InvalidFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.detail or exc.message,
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=exc.detail or exc.message,
        ) from exc
    except Exception as exc:
        logger.error(
            "Upload failed — unexpected error",
            extra={
                "request_id": request_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during upload.",
        ) from exc

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        original_filename=job.original_filename,
        content_type=job.content_type,
        file_size=job.file_size,
        uploaded_at=job.uploaded_at,
        message="Audio uploaded successfully.",
    )
