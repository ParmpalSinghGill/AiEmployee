# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for building wheels and runtime (libpq for asyncpg fallbacks).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt

# Copy source and install the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

# Non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/workspace \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "ai_employee.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
