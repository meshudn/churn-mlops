# syntax=docker/dockerfile:1.7
#
# Multi-stage build for the churn-mlops inference API.
#
# Stage 1 (builder) installs deps into a venv with uv.
# Stage 2 (runtime) copies just the venv + src/ into a slim base — no
# build tools or dev deps in the final image.
#
# Final image: ~ python:3.12-slim + churn-mlops runtime deps + src/.

ARG PYTHON_VERSION=3.12

# ─── builder ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

# Pull in uv from its official image — no curl/wget bootstrap dance.
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /app

# Copy lock + manifest first so the dep layer caches independently of code.
COPY pyproject.toml uv.lock ./

# Copy source; hatch's wheel target is src/ so this is what gets installed.
COPY src ./src

# Install runtime deps only (no dev group). --frozen errors if uv.lock drifts.
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# ─── runtime ─────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

# Run as non-root. uid 10001 keeps it well clear of host numbering collisions.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

# Copy only the artifacts we need to run.
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHURN_API_MLFLOW_TRACKING_URI=file:///app/mlruns

USER appuser

EXPOSE 8000

# Single worker by default — concurrency comes from running more replicas, not
# more workers per process. That keeps the in-memory model object exactly one
# copy per container, which is what the operational model expects.
CMD ["uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
