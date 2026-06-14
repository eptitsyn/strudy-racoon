# syntax=docker/dockerfile:1.7

# Backend image: FastAPI + the `ml` extra (torch + transformers).
# Two stages: build deps with uv, then a slim runtime layer that copies the venv.

FROM python:3.13-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CACHE_DIR=/root/.cache/uv

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# Lockfile-first install for cache reuse on source-only changes.
# BuildKit cache mount keeps uv's download cache across builds so torch /
# transformers wheels don't get re-fetched.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra backend --extra ml

# Now copy sources and install the project itself into the venv.
COPY shared ./shared
COPY detector ./detector
COPY backend ./backend
COPY models.yaml ./models.yaml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra backend --extra ml


FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    MODELS_CONFIG_PATH=/app/models.yaml
    # NOT forcing HF offline: the Llama weights are baked into the image cache
    # below and are used when present (no network), but if a model's weights are
    # missing from the cache transformers will download them from the Hub. Gated
    # repos (e.g. meta-llama) then need HF_TOKEN supplied at runtime — pass it via
    # docker-compose `environment` rather than baking it into the image.

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --shell /sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app /app

# HF cache dir for the `pawn` model's Llama weights. Not baked into the image:
# transformers downloads them from the Hub on first load (gated meta-llama repos
# need HF_TOKEN, supplied via docker-compose). Mount this path as a volume so the
# download persists across container recreations (see docker-compose.yml).
RUN mkdir -p /app/.cache/huggingface/hub && chown -R app:app /app/.cache

USER app

EXPOSE 8000

# Healthcheck targets /health/ready so the container is only "healthy"
# once the default model has finished loading and warming up.
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health/ready || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
