# DispatchOps AI

> AI-powered dispatch intake prototype that converts customer service calls into structured dispatch information using Whisper and LLMs.
# DispatchOps AI

![Python](https://img.shields.io/badge/Python-3.10-blue)

![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green)

![License](https://img.shields.io/badge/License-MIT-yellow)

![Status](https://img.shields.io/badge/Status-Prototype-orange)


DispatchOps AI demonstrates how customer service calls can be transformed into structured dispatch information using modern AI pipelines.

The application accepts customer call recordings, validates and stores audio files, transcribes speech using OpenAI Whisper, and provides an API for extracting structured dispatch information using a Large Language Model.

This project was built as a backend engineering prototype focused on clean architecture, modular design, and AI integration.

---

# Features

## Audio Upload

- Upload customer call recordings (MP3, WAV, M4A)
- File validation
- Secure UUID-based filenames
- Metadata tracking
- Automatic job creation

---

## Speech-to-Text

- OpenAI Whisper integration
- Automatic transcription
- Audio duration detection
- Transcript storage
- Job status tracking

---

## Structured Information Extraction

- LLM-powered extraction pipeline
- Extracts:

  - Customer Name
  - Address
  - Service Issue
  - Trade Category
  - Summary

- Modular design for Groq/OpenAI compatible models

---

## Clean Backend Architecture

```
API
 │
 ▼
Services
 │
 ▼
Repository
 │
 ▼
Storage
```

Each layer has a single responsibility:

- API handles HTTP requests
- Services contain business logic
- Repository manages persistence
- Models define request/response contracts

---

# Tech Stack

## Backend

- Python 3.10
- FastAPI
- Pydantic
- Uvicorn

## AI

- OpenAI Whisper
- LangChain
- Groq LLM

## Utilities

- Structured Logging
- Dependency Injection
- Async FastAPI
- UUID File Storage

---

# Project Structure

```
app/

├── api/
├── services/
├── repositories/
├── models/
├── transcription/
├── extraction/
├── utils/
└── prompts/
```

---

# API Workflow

```
Customer Audio
       │
       ▼
Upload Audio
       │
       ▼
Validate File
       │
       ▼
Create Job
       │
       ▼
Whisper Transcription
       │
       ▼
Transcript
       │
       ▼
LLM Extraction
       │
       ▼
Structured Dispatch Information
```

---


# API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | /api/v1/health/live | Liveness probe |
| GET | /api/v1/health/ready | Readiness probe |
| POST | /api/v1/jobs/upload-audio | Upload customer audio |
| POST | /api/v1/jobs/{job_id}/transcribe | Generate transcript |
| POST | /api/v1/jobs/{job_id}/extract | Extract structured information |

---

# Example Response

```json
{
  "customer_name": "John Smith",
  "address": "24 Green Park Road",
  "issue": "Air conditioner not cooling",
  "trade": "HVAC",
  "summary": "Customer reports AC cooling issue at residential property."
}
```

---

# Local Setup

Clone the repository

```bash
git clone https://github.com/Khushaligupta01/DispatchOpsAI.git
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
uvicorn app.main:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

---

# Current Status

Implemented

- Audio upload
- Audio validation
- Whisper transcription
- Job lifecycle
- REST APIs
- OpenAPI documentation
- Modular backend architecture

Planned Improvements

- PostgreSQL persistence
- Background task queue
- Technician matching algorithm
- Dispatch optimization
- Production observability

---

# Why I Built This

I wanted to explore how modern AI systems can automate the first stage of dispatch operations for skilled trades by converting unstructured customer calls into structured service requests.

The project focuses on backend engineering, AI integration, and clean software architecture rather than frontend development.

---

# License

MIT
