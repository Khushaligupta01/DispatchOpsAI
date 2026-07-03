"""
tests/integration/test_health.py

Integration tests for the health check endpoints.

These tests confirm the full request/response cycle works:
- HTTP request reaches the route
- Route handler executes
- Response matches the expected schema

Why test health endpoints?
- They verify the app factory, router registration, and middleware
  are all wired up correctly.
- If these fail, nothing else will work either.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_returns_200(client: AsyncClient):
    """GET /api/v1/health/live returns HTTP 200."""
    response = await client.get("/api/v1/health/live")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_liveness_response_body(client: AsyncClient):
    """GET /api/v1/health/live returns expected JSON body."""
    response = await client.get("/api/v1/health/live")
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "dispatchops-ai"


@pytest.mark.asyncio
async def test_readiness_returns_200(client: AsyncClient):
    """GET /api/v1/health/ready returns HTTP 200."""
    response = await client.get("/api/v1/health/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_readiness_response_body(client: AsyncClient):
    """GET /api/v1/health/ready returns expected JSON body."""
    response = await client.get("/api/v1/health/ready")
    body = response.json()
    assert body["status"] == "ready"
    assert body["service"] == "dispatchops-ai"
    assert body["version"] == "1.0.0"
