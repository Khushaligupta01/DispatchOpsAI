"""
app/services/upload_service.py

Upload service — handles audio file validation, storage, and job creation.

RESPONSIBILITIES (exactly one per method):
- Validate the uploaded file (type and size).
- Generate a unique, date-organized file path.
- Save the file to disk.
- Create a Job record.
- Store the Job via the repository.
- Return the Job to the caller.

WHY DOES THE SERVICE NOT KNOW ABOUT FASTAPI?
--------------------------------------------
The service receives plain Python values (bytes, string, int).
It never touches Request, UploadFile, or Response objects.
This means it's testable without an HTTP stack — just call the method
with a bytes object and a filename. No TestClient needed.

WHY DOES THE SERVICE NOT KNOW ABOUT STORAGE BACKEND?
-----------------------------------------------------
The service calls self._repo.save(job) — it does not care whether
that's an in-memory dict or a PostgreSQL table. The repository is
injected at construction time. Tests inject a mock repository.
Production injects InMemoryJobRepository now, PostgresJobRepository later.

Interview talking point:
"UploadService has one public method: upload_audio. It validates,
saves to disk, creates a Job, stores it, and returns it. It has no
knowledge of HTTP, no knowledge of the database engine, and no
knowledge of what comes next in the pipeline. Each of those concerns
lives in its own layer."
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from app.repositories.job_repository import AbstractJobRepository
from app.schemas.job import Job, JobStatus
from app.utils.exceptions import DispatchOpsError
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Allowed MIME types for audio uploads
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/x-m4a",
}

# Maximum file size: 20 MB in bytes
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 971 520 bytes


class UploadService:
    """
    Handles audio file upload, validation, and job creation.

    Dependencies are injected via __init__ so they can be swapped in tests.

    Args:
        repository:   Where to persist Job objects.
        upload_dir:   Base directory for saving audio files (from config).
    """

    def __init__(
        self,
        repository: AbstractJobRepository,
        upload_dir: str,
    ) -> None:
        self._repo = repository
        self._upload_dir = Path(upload_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def upload_audio(
        self,
        *,
        file_data: bytes,
        original_filename: str,
        content_type: str,
        request_id: str,
    ) -> Job:
        """
        Validate, save, and register an uploaded audio file as a new Job.

        Args:
            file_data:         Raw bytes of the uploaded file.
            original_filename: The filename provided by the client.
            content_type:      MIME type declared by the client.
            request_id:        Correlation ID for tracing logs back to the request.

        Returns:
            A fully populated Job object with status=UPLOADED.

        Raises:
            InvalidFileTypeError:  Content type is not in ALLOWED_CONTENT_TYPES.
            FileTooLargeError:     File exceeds MAX_FILE_SIZE_BYTES.
            DispatchOpsError:      Any unexpected failure during save or persist.
        """
        logger.info(
            "Upload started",
            extra={
                "request_id": request_id,
                "audio_filename": original_filename,
                "content_type": content_type,
                "file_size": len(file_data),
            },
        )

        # Step 1 — Validate
        self._validate_content_type(content_type, original_filename)
        self._validate_file_size(len(file_data), original_filename)

        # Step 2 — Generate unique path and save to disk
        job_id = str(uuid.uuid4())
        file_path, stored_filename = self._build_file_path(
            job_id=job_id,
            original_filename=original_filename,
        )
        self._save_to_disk(file_data=file_data, file_path=file_path)

        # Step 3 — Create the Job domain object
        now = datetime.now(tz=timezone.utc)
        job = Job(
            job_id=job_id,
            filename=stored_filename,
            original_filename=original_filename,
            content_type=content_type,
            file_size=len(file_data),
            file_path=str(file_path),
            status=JobStatus.UPLOADED,
            uploaded_at=now,
            updated_at=now,
        )

        # Step 4 — Persist via repository
        saved_job = await self._repo.save(job)

        logger.info(
            "Upload completed",
            extra={
                "request_id": request_id,
                "job_id": job_id,
                "audio_filename": stored_filename,
                "file_size": len(file_data),
            },
        )

        return saved_job

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_content_type(self, content_type: str, filename: str) -> None:
        """
        Reject files whose MIME type is not in the allowed set.

        We check the MIME type declared in the Content-Type header, not
        the file extension — extensions can be renamed trivially.
        """
        if content_type not in ALLOWED_CONTENT_TYPES:
            logger.warning(
                "Upload rejected — unsupported file type",
                extra={"content_type": content_type, "audio_filename": filename},
            )
            raise InvalidFileTypeError(
                message=f"Unsupported file type: '{content_type}'",
                detail=(
                    f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}. "
                    f"Received: '{content_type}'."
                ),
            )

    def _validate_file_size(self, size_bytes: int, filename: str) -> None:
        """Reject files that exceed the maximum allowed size."""
        if size_bytes > MAX_FILE_SIZE_BYTES:
            size_mb = size_bytes / (1024 * 1024)
            logger.warning(
                "Upload rejected — file too large",
                extra={
                    "audio_filename": filename,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "limit_mb": 20,
                },
            )
            raise FileTooLargeError(
                message=f"File size {size_mb:.1f} MB exceeds the 20 MB limit.",
                detail=f"Received {size_bytes} bytes. Maximum is {MAX_FILE_SIZE_BYTES} bytes.",
            )

    def _build_file_path(
        self,
        job_id: str,
        original_filename: str,
    ) -> Tuple[Path, str]:
        """
        Build a date-organized, UUID-named file path.

        Structure: uploads/YYYY/MM/DD/<uuid>.<ext>

        Why date-based directories?
        - Prevents a single directory from accumulating thousands of files,
          which degrades filesystem performance.
        - Makes it easy to apply lifecycle policies (delete files older than 30 days).
        - Makes manual auditing human-readable.

        Why UUID filename instead of original_filename?
        - Original filenames can contain path traversal sequences (../../etc/passwd).
        - Original filenames can collide (two clients upload "call.wav").
        - UUIDs are globally unique and safe for filesystem use.
        """
        # Extract file extension from the original filename (lowercase, safe)
        original_path = Path(original_filename)
        extension = original_path.suffix.lower() if original_path.suffix else ".wav"

        # Date-based subdirectory
        today = datetime.now(tz=timezone.utc)
        date_path = self._upload_dir / str(today.year) / f"{today.month:02d}" / f"{today.day:02d}"

        # Stored filename: just the UUID + extension
        stored_filename = f"{job_id}{extension}"
        full_path = date_path / stored_filename

        return full_path, stored_filename

    def _save_to_disk(self, file_data: bytes, file_path: Path) -> None:
        """
        Write the file bytes to disk, creating parent directories as needed.

        Uses Path.mkdir(parents=True, exist_ok=True) so:
        - The full date directory tree is created if it doesn't exist.
        - No error is raised if the directory already exists.
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(file_data)
        except OSError as exc:
            logger.error(
                "Upload failed — could not write file to disk",
                extra={"file_path": str(file_path), "error": str(exc)},
            )
            raise DispatchOpsError(
                message="Failed to save audio file.",
                detail=str(exc),
            ) from exc


# ------------------------------------------------------------------
# Domain-specific exceptions for this service
# ------------------------------------------------------------------

class InvalidFileTypeError(DispatchOpsError):
    """Raised when the uploaded file's MIME type is not supported."""
    pass


class FileTooLargeError(DispatchOpsError):
    """Raised when the uploaded file exceeds the maximum allowed size."""
    pass
