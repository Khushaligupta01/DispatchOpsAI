"""
tests/conftest.py

Shared pytest fixtures for DispatchOps AI tests.

What is conftest.py?
- pytest automatically loads this file before any tests run.
- Fixtures defined here are available to every test in the project
  without needing to import them.

Current fixtures (Feature 1):
- client: An async HTTP test client for the FastAPI app.
- override_settings: Injects test-safe settings (no real API keys needed).

Future fixtures (added with each feature):
- db_session: An async database session connected to a test database.
- mock_groq: A mock that returns a fixed LLM response without hitting the API.
- mock_whisper: A mock that returns a fixed transcript without loading the model.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """
    Async HTTP client for testing FastAPI routes.

    Uses ASGI transport — requests go directly through the app without
    a real network connection. This makes tests fast and self-contained.

    Usage:
        async def test_health(client: AsyncClient):
            response = await client.get("/api/v1/health/live")
            assert response.status_code == 200
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac
