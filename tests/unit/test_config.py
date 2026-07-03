"""
tests/unit/test_config.py

Tests for the configuration module.

These tests verify that Settings loads correctly and that
the lru_cache returns the same instance across calls.
"""

from app.config import Settings, get_settings


def test_get_settings_returns_settings_instance():
    """Settings factory returns a valid Settings object."""
    settings = get_settings()
    assert isinstance(settings, Settings)


def test_get_settings_is_cached():
    """Calling get_settings() twice returns the exact same object."""
    first = get_settings()
    second = get_settings()
    assert first is second


def test_default_values():
    """Default configuration values are set correctly."""
    settings = get_settings()
    assert settings.app_host == "0.0.0.0"
    assert settings.app_port == 8000
    assert settings.groq_model == "llama3-8b-8192"
    assert settings.whisper_model_size == "base"


def test_is_development_flag():
    """is_development returns True when app_env is 'development'."""
    settings = get_settings()
    # In test environment, APP_ENV defaults to 'development'
    if settings.app_env == "development":
        assert settings.is_development is True
