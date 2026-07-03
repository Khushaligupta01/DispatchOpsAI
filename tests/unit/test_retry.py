"""
tests/unit/test_retry.py

Tests for the retry decorator.

These verify that:
- A function that always fails retries the correct number of times.
- A function that fails then succeeds returns the successful result.
- Only specified exception types trigger a retry.
"""

import pytest

from app.utils.retry import retry


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    """Function that fails once then succeeds returns the successful result."""
    call_count = 0

    @retry(max_retries=2, delay_seconds=0)
    async def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("Temporary failure")
        return "success"

    result = await flaky_function()
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_raises_after_max_retries():
    """Function that always fails raises after max_retries + 1 total calls."""
    call_count = 0

    @retry(max_retries=2, delay_seconds=0)
    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise ValueError("Always fails")

    with pytest.raises(ValueError):
        await always_fails()

    assert call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_retry_only_catches_specified_exceptions():
    """Exceptions not in the exceptions tuple are raised immediately."""
    call_count = 0

    @retry(max_retries=3, delay_seconds=0, exceptions=(TypeError,))
    async def raises_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Not a TypeError")

    with pytest.raises(ValueError):
        await raises_value_error()

    # Should fail on first attempt, no retries
    assert call_count == 1
