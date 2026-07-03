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
- [ ] **Feature 2** — Twilio webhook + job creation
- [ ] **Feature 3** — Database layer (PostgreSQL + SQLAlchemy)
- [ ] **Feature 4** — Whisper transcription pipeline
- [ ] **Feature 5** — Groq LLM extraction + confidence scoring
- [ ] **Feature 6** — Technician ranking engine
- [ ] **Feature 7** — Dispatch service + Langfuse tracing
- [ ] **Feature 8** — Job status API
- [ ] **Feature 9** — Tests + documentation
