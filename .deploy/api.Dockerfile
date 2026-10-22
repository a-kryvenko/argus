FROM python:3.12-slim

WORKDIR /var/www/apps/api

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

COPY apps/api/pyproject.toml apps/api/uv.lock ./

RUN uv sync --frozen --no-cache --no-install-local

COPY packages/common /var/www/packages/common

COPY apps/api/src ./src

RUN uv sync --frozen --no-cache
COPY scripts/db/provision.py /var/www/scripts/db/provision.py

ENV PATH="/var/www/apps/api/.venv/bin:$PATH"

EXPOSE 8000

CMD ["api", "serve"]
