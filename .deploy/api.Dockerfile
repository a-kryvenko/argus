FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /var/www/apps/api
WORKDIR /var/www/apps/api

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY apps/api/pyproject.toml apps/api/uv.lock ./

COPY packages/common ./../../packages/common

RUN uv sync --frozen --no-cache

COPY apps/api/app ./app
COPY apps/api/alembic.ini ./alembic.ini
COPY apps/api/alembic ./alembic
COPY scripts/provision-databases.py /var/www/scripts/provision-databases.py
COPY scripts/transfer-databases.py /var/www/scripts/transfer-databases.py

EXPOSE 8000

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
