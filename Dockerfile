# ==================================================
# DispatchOps AI — Dockerfile
#
# Multi-stage build:
# Stage 1 (builder): Installs all dependencies
# Stage 2 (runtime): Copies only what's needed to run
#
# Why multi-stage?
# - The final image doesn't include build tools (gcc, pip cache, etc.)
# - Smaller image = faster deploys, smaller attack surface
# - Clean separation between build-time and run-time concerns
#
# Interview talking point:
# "We use a multi-stage build. The builder stage compiles everything,
# the runtime stage copies only the installed packages. This keeps
# the production image lean without any build tooling in it."
# ==================================================

# --- Stage 1: Builder ---
FROM python:3.10-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies required to build Python packages
# (psycopg2, numpy, and others need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching.
# If requirements.txt hasn't changed, Docker reuses the cached layer
# and skips re-installing packages — much faster rebuilds.
COPY requirements.txt .

# Install Python dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.10-slim AS runtime

WORKDIR /app

# Install only the runtime system dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the application source code
COPY . .

# Create the audio uploads directory
RUN mkdir -p uploads/audio

# Run as a non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the application port
EXPOSE 8000

# Default command: start the FastAPI app with uvicorn
# --reload is for dev only; override this in docker-compose for production
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
