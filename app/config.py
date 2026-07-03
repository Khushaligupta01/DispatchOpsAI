"""
app/config.py

Central configuration module for DispatchOps AI.

All environment variables are loaded here using Pydantic Settings.
This is the SINGLE source of truth for configuration across the entire app.

Why Pydantic Settings?
- Automatically reads from .env files and environment variables.
- Validates types at startup — if DATABASE_URL is missing, the app refuses to start.
- Provides IDE autocompletion for every config value.
- Makes it easy to swap values between environments (dev/prod) without code changes.

Interview talking point:
"All config is centralized in one Settings class. If a required variable is missing,
the app fails fast at startup with a clear error rather than failing silently at runtime."
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Pydantic validates every field on instantiation. If a required field
    is missing or has the wrong type, the app raises a ValidationError
    immediately — not 10 minutes later when that code path is hit.
    """

    model_config = SettingsConfigDict(
        env_file=".env",          # Load from .env file in the project root
        env_file_encoding="utf-8",
        case_sensitive=False,     # APP_HOST and app_host are equivalent
        extra="ignore",           # Ignore unknown env vars — keeps config clean
    )

    # --- Application ---
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"

    # --- PostgreSQL ---
    database_url: str

    # --- Redis ---
    redis_url: str = "redis://redis:6379/0"

    # --- Celery ---
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"

    # --- Whisper ---
    whisper_model_size: str = "base"

    # --- Langfuse ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Audio Storage ---
    audio_upload_dir: str = "uploads/audio"

    @property
    def is_development(self) -> bool:
        """Convenience flag — use this instead of comparing strings."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    The @lru_cache decorator ensures Settings() is only instantiated once
    for the lifetime of the application, no matter how many times
    get_settings() is called. This avoids re-reading the .env file on
    every request.

    Usage:
        from app.config import get_settings
        settings = get_settings()
        print(settings.groq_model)
    """
    return Settings()
