"""
app/api/v1/router.py

API v1 route aggregator.

This file is the single place where all v1 routers are registered.
main.py includes this one router — it doesn't need to know about individual
route files. Adding a new feature means adding one line here.

Why version the API (/v1/)?
- If you need to make a breaking change later (v2), existing clients keep
  working on v1 until they migrate.
- It's a standard practice that interviewers expect to see.
"""

from fastapi import APIRouter

from app.api.v1 import health, jobs, transcription

# The main v1 router — all sub-routers are included here
api_router = APIRouter(prefix="/api/v1")

# Feature 1: Health checks
api_router.include_router(health.router)

# Feature 2: Audio upload and job creation
api_router.include_router(jobs.router)

# Feature 3: Whisper transcription
api_router.include_router(transcription.router)

# Future features will be added here:
# api_router.include_router(webhooks.router)    # Twilio webhook
# api_router.include_router(technicians.router) # Technician roster
# api_router.include_router(dispatch.router)    # Dispatch history
