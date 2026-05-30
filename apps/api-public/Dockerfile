FROM python:3.12-slim

WORKDIR /workspace/app/api/

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY apps/api-public/pyproject.toml apps/api-public/uv.lock ./

COPY packages/common ./../../packages/common
COPY packages/forecast-core ./../../packages/forecast-core

RUN uv sync --frozen --no-cache

COPY apps/api-public/app ./app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]