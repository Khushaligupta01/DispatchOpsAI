"""
app/api/v1/extraction.py

LLM extraction route for DispatchOps AI.

Single endpoint:
    POST /api/v1/jobs/{job_id}/extract

Route responsibility: HTTP adapter only.
- Extract job_id from the path.
- Call ExtractionService.
- Map domain exceptions to HTTP status codes.
- Return ExtractionResponse.

Zero business logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.job import ExtractionResponse
from app.services.extraction_service import (
    ExtractionService,
    TranscriptMissingError,
)
from app.utils.exceptions import ExtractionError, JobNotFoundError
from app.utils.logger import get_logger

from .dependencies import get_extraction_service

logger = get_logger(__name__)

router = APIRouter(prefix="/jobs", tags=["Extraction"])


@router.post(
    "/{job_id}/extract",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract structured dispatch information from a job transcript",
    description=(
        "Sends the job transcript to the Groq LLM and extracts structured "
        "dispatch information: customer name, address, issue, trade, and summary. "
        "The job must have been transcribed first (status=TRANSCRIBED)."
    ),
    responses={
        200: {"description": "Extraction completed successfully"},
        400: {"description": "Job has no transcript — call /transcribe first"},
        404: {"description": "Job not found"},
        500: {"description": "LLM call failed or returned invalid JSON"},
    },
)
async def extract_job(
    job_id: str,
    extraction_service: ExtractionService = Depends(get_extraction_service),
) -> ExtractionResponse:
    """
    Extract structured dispatch information from the job's transcript.

    Advances job status: TRANSCRIBED → EXTRACTING → EXTRACTED.
    Returns the nested extraction object on success.
    """
    logger.info("Extract request received", extra={"job_id": job_id})

    try:
        job = await extraction_service.extract_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.message,
        ) from exc
    except TranscriptMissingError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc
    except ExtractionError as exc:
        logger.error(
            "Extraction endpoint — LLM error",
            extra={"job_id": job_id, "error": exc.message},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc.message,
        ) from exc
    except Exception as exc:
        logger.error(
            "Extraction endpoint — unexpected error",
            extra={
                "job_id": job_id,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during extraction.",
        ) from exc

    # job.extraction and job.extracted_at are guaranteed non-None
    # when status is EXTRACTED — the service contract ensures this.
    return ExtractionResponse(
        job_id=job.job_id,
        status=job.status,
        extraction=job.extraction,      # type: ignore[arg-type]
        extracted_at=job.extracted_at,  # type: ignore[arg-type]
        message="Extraction completed successfully.",
    )
