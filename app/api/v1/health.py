"""
app/api/v1/health.py

Health check endpoints for DispatchOps AI.

Why health endpoints?
- Docker Compose and any orchestrator (ECS, Railway, Render) uses /health
  to decide whether the container is alive and ready to receive traffic.
- Without a health endpoint, a container that starts but crashes silently
  will still receive traffic.

Two endpoints:
- /health/live  — "Is the process running?" Simple 200. Used by Docker.
- /health/ready — "Is the app ready to serve requests?" Checks dependencies.
                  Used to hold traffic during startup.

Interview talking point:
"Liveness and readiness are different things. Liveness says the process hasn't
crashed. Readiness says all dependencies are connected and the app can actually
do useful work. You don't want traffic sent to a container that's still
connecting to the database."
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["Health"])


class LivenessResponse(BaseModel):
    """Response model for the liveness probe."""
    status: str
    service: str


class ReadinessResponse(BaseModel):
    """Response model for the readiness probe."""
    status: str
    service: str
    version: str


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Returns 200 if the application process is running. "
                "Used by Docker and load balancers to detect crashes.",
)
async def liveness() -> LivenessResponse:
    """
    Liveness check — proves the process is alive.

    This endpoint does the minimum possible work. It should never fail
    unless the Python process itself is broken.
    """
    return LivenessResponse(status="ok", service="dispatchops-ai")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Returns 200 if the application is ready to handle requests. "
                "Will expand to check DB and Redis connectivity in later features.",
)
async def readiness() -> ReadinessResponse:
    """
    Readiness check — proves the app is fully initialized.

    In Feature 1 this just confirms the app started. In later features,
    this will verify PostgreSQL and Redis connectivity before returning 200.
    """
    return ReadinessResponse(
        status="ready",
        service="dispatchops-ai",
        version="1.0.0",
    )
