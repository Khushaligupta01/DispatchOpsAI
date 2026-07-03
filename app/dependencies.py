"""
app/dependencies.py

FastAPI dependency injection for DispatchOps AI.

What is dependency injection in FastAPI?
- Instead of importing services directly inside route handlers, you declare
  what a route needs as a function parameter with Depends().
- FastAPI resolves the dependency and injects it automatically.
- This makes routes easier to test: you can override dependencies in tests
  without changing any production code.

Right now this only has the settings dependency.
Future features will add:
- get_db() — async database session per request
- get_langfuse() — Langfuse tracing client

Interview talking point:
"FastAPI's dependency injection means route handlers never import services
directly. In tests I override get_db() with a test database session
using app.dependency_overrides — no mocking frameworks needed."
"""

from collections.abc import AsyncGenerator

from fastapi import Depends

from app.config import Settings, get_settings


async def get_app_settings(
    settings: Settings = Depends(get_settings),
) -> Settings:
    """
    Dependency that provides the application settings to a route.

    Usage in a route:
        @router.get("/example")
        async def example(settings: Settings = Depends(get_app_settings)):
            return {"model": settings.groq_model}
    """
    return settings


# --- Placeholder for database session dependency ---
# This will be implemented in Feature 3 (Database layer).
#
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as session:
#         try:
#             yield session
#         finally:
#             await session.close()
