# DispatchOps AI

**Asynchronous AI Dispatch & Revenue Optimization Engine for Skilled Trades**

DispatchOps AI automatically handles the entire dispatch workflow when a customer calls a service company (HVAC, Plumbing, Electrical, Roofing):

1. Receives the customer call via **Twilio**
2. Downloads the call recording
3. Transcribes the audio using **Whisper**
4. Extracts structured job details using **Groq LLM**
5. Scores available technicians using **Pandas/NumPy**
6. Dispatches the best-matched technician
7. Tracks every AI step with **Langfuse**

Everything runs asynchronously via **Redis + Celery**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Pydantic v2 |
| Task Queue | Celery + Redis |
| Database | PostgreSQL + SQLAlchemy Async |
| Transcription | OpenAI Whisper (local) |
| LLM | Groq API + LangChain |
| Telephony | Twilio |
| Observability | Langfuse |
| Data Processing | Pandas + NumPy |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-asyncio |

---

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/yourname/dispatchops-ai
cd dispatchops-ai

# 2. Copy environment template
cp .env.example .env
# Fill in your API keys in .env

# 3. Start all services
docker-compose up --build

# 4. Verify the app is running
curl http://localhost:8000/api/v1/health/live
# {"status":"ok","service":"dispatchops-ai"}

# 5. Open the API docs
# http://localhost:8000/docs
```

---

## Running Tests

```bash
# Install dependencies locally (for running tests outside Docker)
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

---

## Project Structure

```
dispatchops-ai/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── config.py            # All environment variables (Pydantic Settings)
│   ├── dependencies.py      # FastAPI dependency injection
│   ├── api/v1/              # HTTP routes (no business logic)
│   ├── services/            # Business logic layer
│   ├── repositories/        # Database queries
│   ├── workers/             # Celery tasks
│   ├── transcription/       # Whisper speech-to-text
│   ├── llm/                 # Groq client + prompt templates
│   ├── ranking/             # Technician scoring engine
│   ├── integrations/        # Twilio + Langfuse clients
│   ├── db/                  # SQLAlchemy models + session
│   ├── schemas/             # Pydantic request/response models
│   └── utils/               # Logger, exceptions, retry
├── tests/
├── uploads/audio/           # Downloaded call recordings
└── docs/
```

---

## System Flow

```
Customer Call → Twilio → FastAPI Webhook → Redis Queue
    → Celery Worker → Whisper Transcription
    → Groq Extraction → Confidence Check
    → (>= 0.7) Technician Ranking → Dispatch → PostgreSQL
    → (< 0.7) Mark for Human Review
    → Langfuse traces every step
```

---

## Features (Incremental Build)

- [x] **Feature 1** — Project scaffold, config, logging, Docker
- [x] **Feature 2** — Audio upload API, job creation, repository pattern
- [ ] **Feature 3** — Celery worker + Redis task queue
- [ ] **Feature 4** — Whisper transcription pipeline
- [ ] **Feature 5** — PostgreSQL + SQLAlchemy (replaces in-memory repository)
- [ ] **Feature 6** — Groq LLM extraction + confidence scoring
- [ ] **Feature 7** — Technician ranking engine
- [ ] **Feature 8** — Dispatch service + Langfuse tracing
- [ ] **Feature 9** — Job status API + tests

---

## API Endpoints

### POST /api/v1/jobs/upload-audio

Upload a customer call recording. Returns a `job_id` to track the pipeline.

**Supported formats:** `audio/wav`, `audio/mpeg`, `audio/mp3`, `audio/x-m4a`
**Maximum size:** 20 MB

**Example request (curl):**
```bash
curl -X POST http://localhost:8000/api/v1/jobs/upload-audio \
  -F "file=@customer_call.wav;type=audio/wav"
```

**Example response (201 Created):**
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "UPLOADED",
  "filename": "3fa85f64-5717-4562-b3fc-2c963f66afa6.wav",
  "original_filename": "customer_call.wav",
  "content_type": "audio/wav",
  "file_size": 184320,
  "uploaded_at": "2026-07-03T20:00:00Z",
  "message": "Audio uploaded successfully."
}
```

**Error responses:**
- `400` — Unsupported file type
- `413` — File exceeds 20 MB
- `500` — Unexpected server error

**Upload storage:** Files are saved to `uploads/YYYY/MM/DD/<uuid>.<ext>`

### GET /api/v1/health/live
Liveness probe — returns 200 if the process is running.

### GET /api/v1/health/ready
Readiness probe — returns 200 if the app is ready to serve requests.
