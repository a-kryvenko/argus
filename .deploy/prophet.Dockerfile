FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /var/www/apps/prophet
COPY packages/common /var/www/packages/common
COPY packages/forecast /var/www/packages/forecast
COPY packages/forecast-core /var/www/packages/forecast-core
COPY apps/prophet/pyproject.toml apps/prophet/uv.lock ./
COPY apps/prophet/src ./src
RUN uv sync --frozen --no-cache
ENV PATH="/var/www/apps/prophet/.venv/bin:$PATH"
CMD ["prophet", "worker"]
