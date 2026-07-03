"""
app/utils/retry.py

Retry decorator with exponential backoff for external API calls.

Why retry logic?
- External APIs (Groq, Twilio) are not 100% reliable. A single transient
  network error should not fail an entire job pipeline.
- Exponential backoff avoids hammering a struggling API — each retry waits
  longer than the last.
- max_retries caps the total attempts so a permanently failed call doesn't
  loop forever.

Why a decorator?
- Clean. Add @retry() above any function that calls an external service.
- No try/except boilerplate scattered across services.
- Easy to test — just mock the decorated function.

Interview talking point:
"External calls to Groq and Twilio are wrapped in retry with exponential backoff.
A transient 429 or timeout gets retried transparently. After max retries,
we raise the original exception and let the Celery task handle it."
"""

from __future__ import annotations  # Enables X | Y union syntax on Python 3.10

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

from app.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function on failure with exponential backoff.

    Works for both sync and async functions.

    Args:
        max_retries:    Maximum number of retry attempts (not counting first call).
        delay_seconds:  Initial wait time between retries in seconds.
        backoff_factor: Multiplier applied to delay after each retry.
                        With delay=1.0 and backoff=2.0: waits 1s, 2s, 4s...
        exceptions:     Tuple of exception types to catch and retry on.
                        Default catches all exceptions.

    Example:
        @retry(max_retries=3, delay_seconds=1.0, exceptions=(httpx.HTTPError,))
        async def call_groq_api(prompt: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            current_delay = delay_seconds

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        logger.warning(
                            "Retrying after failure",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay_seconds": current_delay,
                                "error": str(exc),
                            },
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            "Max retries exceeded",
                            extra={
                                "function": func.__name__,
                                "attempts": max_retries + 1,
                                "error": str(exc),
                            },
                        )

            raise last_exception  # type: ignore[misc]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            current_delay = delay_seconds

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        logger.warning(
                            "Retrying after failure",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay_seconds": current_delay,
                                "error": str(exc),
                            },
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(
                            "Max retries exceeded",
                            extra={
                                "function": func.__name__,
                                "attempts": max_retries + 1,
                                "error": str(exc),
                            },
                        )

            raise last_exception  # type: ignore[misc]

        # Return the correct wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
