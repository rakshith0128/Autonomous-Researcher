# syntax=docker/dockerfile:1
#
# Multi-stage build: Node compiles the React bundle, Python serves it alongside
# the API from a single process on a single port. One container, one URL, no
# CORS, one thing to keep warm on a free tier.
#
# Targets Hugging Face Spaces (Docker SDK), which requires listening on 7860
# and running as a non-root uid 1000.

# ---------------------------------------------------------------- stage 1: UI
FROM node:20-alpine AS frontend

WORKDIR /build

# Manifests first so `npm ci` is cached until dependencies actually change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ------------------------------------------------------------ stage 2: server
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl is here for the container healthcheck only. Everything else in the
# dependency tree ships manylinux wheels, so there is no compiler in this image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Dependencies before source: a code change must not reinstall scipy.
COPY --chown=user requirements.txt ./
RUN pip install --user -r requirements.txt

COPY --chown=user backend/ ./backend/
COPY --chown=user --from=frontend /build/dist ./frontend/dist

# Writable at runtime for the SQLite ledger, Chroma index, and run artefacts.
ENV DATA_DIR=$HOME/app/data
RUN mkdir -p "$DATA_DIR"

# fastembed downloads its ONNX embedding model on first use. Doing that at
# build time means the first run does not stall on a model download while a
# reviewer watches an empty progress feed.
ENV FASTEMBED_CACHE_PATH=$HOME/app/.fastembed
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); \
print('embedding model cached')" || echo "WARN: embedding model not pre-cached; will download on first use"

# 7860 is the default because Hugging Face Spaces requires it. Railway and
# Render instead inject $PORT and route to whatever it names, so the port is
# read from the environment with 7860 as the fallback -- one image, three hosts.
ENV PORT=7860
EXPOSE 7860

# Shell form deliberately: the exec form does not expand ${PORT}, so an exec-form
# CMD would have the server listen on the literal string and the platform's
# health check would never pass.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/api/health" || exit 1

CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860} --timeout-keep-alive 120

