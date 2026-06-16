FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
    
RUN mkdir -p /var/www/apps/api
WORKDIR /var/www/apps/api

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY apps/api/pyproject.toml apps/api/uv.lock ./

COPY packages/common ./../../packages/common
COPY packages/forecast ./../../packages/forecast

RUN mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=ssh uv sync --frozen --no-cache

COPY apps/api/app ./app
COPY apps/api/src ./src

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]