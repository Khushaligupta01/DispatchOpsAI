"""
app/workers/tasks.py

Celery task definitions for the DispatchOps AI pipeline.

This module defines the background tasks that run after a Twilio webhook
is received. Each task represents one stage of the AI pipeline.

Current state (Feature 1):
- Only a placeholder task exists to confirm Celery is wired up correctly.
- Real pipeline tasks will be added in Features 2-6.

Pipeline tasks (to be implemented):
1. process_call_pipeline — orchestrates the full pipeline for one call
2. download_recording    — downloads audio from Twilio
3. transcribe_audio      — runs Whisper on the audio file
4. extract_job_details   — sends transcript to Groq for structured extraction
5. rank_and_dispatch     — scores technicians and saves dispatch

Interview talking point:
"Each pipeline stage is a separate Celery task. If extraction fails, only
that stage retries — not the entire pipeline from scratch. Task results
are stored in Redis so the API can report real-time pipeline status."
"""

from app.utils.logger import get_logger
from app.workers.celery_worker import celery_app

logger = get_logger(__name__)


@celery_app.task(
    name="tasks.health_check",
    bind=True,
    max_retries=0,
)
def health_check_task(self) -> dict:  # type: ignore[type-arg]
    """
    Placeholder task to verify Celery and Redis are connected.

    This task will be removed in a later feature. It exists only
    to confirm the worker starts and can process tasks.

    Returns:
        A dict with status confirmation.
    """
    logger.info("Celery health check task executed")
    return {"status": "ok", "worker": "dispatchops_worker"}


# --- Future tasks (to be implemented in later features) ---
#
# @celery_app.task(name="tasks.process_call_pipeline", bind=True, max_retries=3)
# def process_call_pipeline(self, job_id: str, recording_url: str) -> dict:
#     """Full AI pipeline for one customer call."""
#     ...
