# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv
WORKDIR /app

FROM base AS builder
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM builder AS dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
COPY alembic.ini ./
COPY tests ./tests
ENV PATH="/app/.venv/bin:$PATH"
CMD ["pytest"]

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/models
RUN useradd --create-home --uid 1000 app && mkdir /models && chown app:app /models
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app corpus ./corpus
COPY --chown=app:app eval ./eval
USER app
EXPOSE 8000
CMD ["uvicorn", "agro_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
