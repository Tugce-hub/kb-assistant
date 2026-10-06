# One image, three entrypoints (api / slack / mcp). Models are baked in at build
# time so pods start without reaching Hugging Face and cold start stays short.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/opt/models HF_HUB_DISABLE_SYMLINKS_WARNING=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY config ./config
# Pre-download the embedding + reranker models named in config/settings.yaml.
RUN python -c "from kbassist.config import get_settings as g; s=g(); \
from kbassist.index.embeddings import _embedder, _cross_encoder; \
_embedder(s.get('embedding.model')); _cross_encoder(s.get('reranker.model'))"

RUN mkdir -p /app/data && chown -R app /app/data /opt/models
USER app

EXPOSE 8080
CMD ["uvicorn", "kbassist.api:app", "--host", "0.0.0.0", "--port", "8080"]
