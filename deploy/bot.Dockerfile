# syntax=docker/dockerfile:1.7

# Bot image: aiogram + httpx + tenacity. No torch, no transformers — keeps
# the layer small (~80MB) and rebuilds fast.

FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CACHE_DIR=/root/.cache/uv

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra bot

COPY shared ./shared
COPY bot ./bot
# Bot reuses backend.logging_config; copy it but no need for routers/services.
COPY backend/__init__.py backend/logging_config.py ./backend/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra bot


FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --shell /sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

USER app

CMD ["python", "-m", "bot.main"]
