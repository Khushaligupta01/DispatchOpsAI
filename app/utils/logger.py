"""
app/utils/logger.py

Structured logging for DispatchOps AI.

Why structured (JSON) logging?
- Plain text logs are hard to search and filter in production.
- JSON logs can be ingested by log aggregators (Datadog, CloudWatch, Loki)
  and queried with structured filters like: level=ERROR AND service=ranking.
- Every log entry carries a consistent set of fields: timestamp, level,
  service name, and message. No more grep-and-hope.

Why not just use print()?
- print() has no log levels (you can't silence debug output in prod).
- print() has no structure (can't filter by field).
- print() doesn't flush reliably in containerized environments.

Interview talking point:
"We use structured JSON logging so every log entry is machine-readable.
In production you'd ship these to a log aggregator and set up alerts on
ERROR-level events without changing a single line of application code."
"""

import logging
import sys

from pythonjsonlogger import jsonlogger


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a configured logger instance.

    Each module gets its own logger with the module name as the identifier.
    This means log entries show exactly which module produced them.

    Args:
        name: The logger name, typically __name__ from the calling module.

    Returns:
        A configured Logger instance that outputs structured JSON.

    Usage:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Job created", extra={"job_id": "abc123", "job_type": "HVAC"})
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    # JSON formatter — each log line is valid JSON with standard fields
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Stream handler — outputs to stdout (captured by Docker / log aggregators)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # Set log level from the application config
    # We import here (not at module top) to avoid circular imports at startup
    from app.config import get_settings

    settings = get_settings()
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent log messages from propagating to the root logger
    # (avoids duplicate output)
    logger.propagate = False

    return logger
