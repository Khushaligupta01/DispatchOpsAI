"""
app/repositories/job_repository.py

Repository interface and in-memory implementation for Job persistence.

WHY A REPOSITORY INTERFACE?
----------------------------
The service layer (UploadService) needs to save and retrieve jobs.
But right now we have no database — and we don't want to block Feature 2
on setting up PostgreSQL.

The solution: define an abstract interface (AbstractJobRepository) that
describes WHAT operations are available, then provide a concrete
implementation (InMemoryJobRepository) that stores jobs in a plain dict.

In Feature 5, we add PostgreSQL by writing PostgresJobRepository that
implements the same interface. The service layer never changes — we just
swap which implementation is injected via FastAPI's dependency system.

This is the Repository Pattern. It decouples business logic from storage.

WHY ABSTRACT BASE CLASS?
------------------------
Python's abc.ABC + @abstractmethod means:
- If you create a subclass but forget to implement a method, Python raises
  TypeError at import time — not during a live request.
- The interface is self-documenting. Any engineer reading the codebase knows
  exactly what methods any repository must provide.

Interview talking point:
"I defined an AbstractJobRepository with four methods: save, get_by_id,
get_all, and update_status. Feature 2 uses an in-memory dict. Feature 5
swaps it to PostgreSQL. The UploadService never changes — only the injected
repository implementation changes. This is the repository pattern, and it
made the PostgreSQL migration a two-hour task instead of a two-day rewrite."
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional

from app.schemas.job import Job, JobStatus
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AbstractJobRepository(abc.ABC):
    """
    Abstract interface that every job repository must implement.

    Defines the contract between the service layer and the storage layer.
    The service layer depends on this interface — never on a concrete class.
    """

    @abc.abstractmethod
    async def save(self, job: Job) -> Job:
        """
        Persist a new job and return it.

        Args:
            job: The Job object to save.

        Returns:
            The saved Job (may include DB-generated fields in later implementations).

        Raises:
            DispatchOpsError: If the job cannot be saved.
        """
        ...

    @abc.abstractmethod
    async def get_by_id(self, job_id: str) -> Optional[Job]:
        """
        Retrieve a job by its unique identifier.

        Args:
            job_id: The UUID string of the job.

        Returns:
            The Job if found, None if not.
        """
        ...

    @abc.abstractmethod
    async def get_all(self) -> List[Job]:
        """
        Retrieve all stored jobs.

        Returns:
            List of all Job objects. Empty list if none exist.
        """
        ...

    @abc.abstractmethod
    async def update_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        """
        Update the status of an existing job.

        Used by Celery tasks in later features to advance pipeline state.

        Args:
            job_id: The UUID string of the job.
            status: The new JobStatus to set.

        Returns:
            The updated Job if found, None if the job_id doesn't exist.
        """
        ...


class InMemoryJobRepository(AbstractJobRepository):
    """
    In-memory implementation of AbstractJobRepository.

    Stores jobs in a plain Python dictionary keyed by job_id.
    Data is lost when the process restarts — this is intentional for
    Feature 2. PostgreSQL persistence is added in Feature 5.

    Why a dict instead of a list?
    - O(1) lookup by job_id vs O(n) scan through a list.
    - Mirrors how a database uses a primary key index.

    Thread safety note:
    - For this project, async concurrency (multiple requests at once) is
      handled by FastAPI's event loop. Since Python's GIL prevents true
      parallel dict writes within one process, the dict is safe here.
    - A production in-memory store would use asyncio.Lock for correctness.
    """

    def __init__(self) -> None:
        # The in-memory store: {job_id: Job}
        self._store: Dict[str, Job] = {}
        logger.info("InMemoryJobRepository initialized")

    async def save(self, job: Job) -> Job:
        """Save a job to the in-memory store."""
        self._store[job.job_id] = job
        logger.info(
            "Job saved to memory store",
            extra={"job_id": job.job_id, "status": job.status},
        )
        return job

    async def get_by_id(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by ID from the in-memory store."""
        job = self._store.get(job_id)
        if job is None:
            logger.debug("Job not found", extra={"job_id": job_id})
        return job

    async def get_all(self) -> List[Job]:
        """Return all jobs as a list."""
        return list(self._store.values())

    async def update_status(self, job_id: str, status: JobStatus) -> Optional[Job]:
        """Update a job's status in the in-memory store."""
        job = self._store.get(job_id)
        if job is None:
            logger.warning(
                "Cannot update status — job not found",
                extra={"job_id": job_id, "requested_status": status},
            )
            return None

        # Pydantic models are immutable by default — create an updated copy
        updated_job = job.model_copy(update={"status": status})
        self._store[job_id] = updated_job

        logger.info(
            "Job status updated",
            extra={"job_id": job_id, "new_status": status},
        )
        return updated_job
