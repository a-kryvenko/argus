FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /var/www/apps/api
WORKDIR /var/www/apps/api

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY apps/api/pyproject.toml apps/api/uv.lock ./

# Keep third-party dependencies cached when application or local package code changes.
RUN uv sync --frozen --no-cache --no-install-local

COPY packages/common ./../../packages/common

COPY apps/api/src ./src

RUN uv sync --frozen --no-cache
COPY scripts/db/provision.py /var/www/scripts/db/provision.py

ENV PATH="/var/www/apps/api/.venv/bin:$PATH"

EXPOSE 8000

CMD [".venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
