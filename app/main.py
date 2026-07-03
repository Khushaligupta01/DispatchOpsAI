"""
app/main.py

FastAPI application factory for DispatchOps AI.

This is the entry point for the web server. It:
1. Creates the FastAPI application instance.
2. Registers startup and shutdown lifecycle events.
3. Adds middleware (CORS for now — more in later features).
4. Registers all API routes.
5. Adds a global exception handler so unhandled errors return clean JSON.

Why a factory function (create_app)?
- Testable: tests can call create_app() to get a fresh app instance
  without importing a module-level singleton.
- Flexible: different configurations can be passed for different environments.
- Explicit: you can see everything that's wired up in one place.

Interview talking point:
"The app is created by a factory function. This makes it easy to create
isolated instances in tests and keeps the startup sequence explicit and readable."
"""

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown events.

    The 'async with' pattern (lifespan) is the modern FastAPI approach —
    it replaces the older @app.on_event("startup") decorator.

    Everything before 'yield' runs on startup.
    Everything after 'yield' runs on shutdown.

    Right now this just logs startup. Future features will:
    - Initialize the database connection pool
    - Verify Redis connectivity
    - Pre-load the Whisper model in the worker
    """
    # --- Startup ---
    logger.info(
        "DispatchOps AI starting up",
        extra={
            "environment": settings.app_env,
            "log_level": settings.app_log_level,
        },
    )

    yield  # App is running and serving requests here

    # --- Shutdown ---
    logger.info("DispatchOps AI shutting down")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI instance.

    Returns:
        A fully configured FastAPI application ready to serve requests.
    """
    app = FastAPI(
        title="DispatchOps AI",
        description=(
            "Asynchronous AI Dispatch & Revenue Optimization Engine for Skilled Trades. "
            "Automatically transcribes customer calls, extracts job details using an LLM, "
            "ranks available technicians, and dispatches the best match."
        ),
        version="1.0.0",
        # Disable docs in production to avoid exposing API structure
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # --- Middleware ---
    # CORS: Allow any origin in development. Lock this down in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(api_router)

    # --- Global Exception Handler ---
    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch any unhandled exception and return a clean JSON error response.

        Without this, FastAPI returns a generic 500 with an HTML page.
        With this, every error returns a consistent JSON envelope that
        clients can reliably parse.
        """
        logger.error(
            "Unhandled exception",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "detail": str(exc) if settings.is_development else None,
            },
        )

    logger.info("FastAPI application created", extra={"routes": len(app.routes)})
    return app


# Module-level app instance — used by uvicorn and tests
app = create_app()
