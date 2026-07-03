"""
app/api/v1/dependencies.py

Route-level FastAPI dependency providers.

This file wires concrete implementations to the abstract interfaces
that services depend on. It is the composition root for the API layer.

WHY IS THIS SEPARATE FROM app/dependencies.py?
----------------------------------------------
app/dependencies.py holds application-wide dependencies (DB session, settings).
This file holds route-specific dependencies (which service implementation
to inject for a given router).

As the project grows, each feature area can have its own dependency
providers without polluting the global dependency file.

WHY A MODULE-LEVEL SINGLETON FOR THE REPOSITORY?
-------------------------------------------------
InMemoryJobRepository stores state in a dict. If we created a new instance
on every request, every request would start with an empty store — uploads
from previous requests would vanish.

Using a module-level instance means all requests share the same dict.
When Feature 5 replaces this with PostgresJobRepository, the singleton
pattern is no longer needed (the database persists state externally),
but the interface stays identical.

Interview talking point:
"The dependency provider is the only place that knows which concrete
repository is in use. Swapping from in-memory to PostgreSQL is a
one-line change here. The service, the route, and the tests never change."
"""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.repositories.job_repository import InMemoryJobRepository
from app.services.upload_service import UploadService

# --- Shared repository instance ---
# Module-level singleton so all requests share the same in-memory store.
# Replaced with a database-backed repository in Feature 5.
_job_repository = InMemoryJobRepository()


@lru_cache
def get_upload_service() -> UploadService:
    """
    Dependency provider for UploadService.

    FastAPI calls this function and injects the returned service
    into any route that declares: upload_service: UploadService = Depends(get_upload_service)

    The @lru_cache ensures the same service instance is reused across
    requests — important because the service holds a reference to the
    shared repository instance.

    Returns:
        A configured UploadService ready to handle uploads.
    """
    settings = get_settings()
    return UploadService(
        repository=_job_repository,
        upload_dir=settings.audio_upload_dir,
    )
