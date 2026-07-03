"""
app/workers/celery_worker.py

Celery application factory for DispatchOps AI.

What is Celery?
- Celery is a distributed task queue. It lets you run functions
  asynchronously in a separate process (the worker).
- When Twilio sends a webhook, the FastAPI handler takes less than 200ms
  to enqueue a Celery task and return 200 to Twilio.
- The Celery worker then picks up the task and runs the full AI pipeline
  (Whisper + Groq + ranking) in the background — which can take 5-30 seconds.

Why Redis as the broker?
- Redis is already in the stack. Using it as a Celery broker avoids adding
  another service (like RabbitMQ).
- Redis lists work perfectly as a FIFO task queue.
- The result backend (Redis DB 1) stores task status so the API can report
  whether a pipeline is pending, running, or complete.

Interview talking point:
"The webhook returns immediately. All AI processing happens in the Celery
worker. This means call volume spikes don't slow down the Twilio webhook
response — they just grow the Redis queue, and we can scale workers
independently to drain it."
"""

from celery import Celery

from app.config import get_settings

settings = get_settings()

# Create the Celery application instance
# The first argument is the name of the current module — used for naming tasks
celery_app = Celery(
    "dispatchops",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# --- Celery Configuration ---
celery_app.conf.update(
    # Serialize tasks as JSON (readable and language-agnostic)
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task retry defaults — individual tasks can override these
    task_max_retries=3,
    task_default_retry_delay=5,  # seconds

    # Result expiry — task results are deleted from Redis after 1 hour
    # This prevents unbounded Redis memory growth
    result_expires=3600,

    # Track task start time — useful for measuring pipeline duration
    task_track_started=True,

    # Worker configuration
    worker_prefetch_multiplier=1,  # Process one task at a time per worker thread
                                    # Prevents one slow task from blocking others
)

# Auto-discover tasks in the workers/tasks.py module
# When the worker starts, Celery will find and register all @celery_app.task functions
celery_app.autodiscover_tasks(["app.workers"])
