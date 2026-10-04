FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /var/www/apps/prophet
COPY apps/prophet/pyproject.toml apps/prophet/uv.lock ./

# Keep third-party dependencies cached when application or local package code changes.
RUN uv sync --frozen --no-cache --no-install-local

COPY packages/common /var/www/packages/common
COPY packages/forecast /var/www/packages/forecast
COPY packages/forecast-core /var/www/packages/forecast-core
COPY apps/prophet/src ./src
RUN uv sync --frozen --no-cache
ENV PATH="/var/www/apps/prophet/.venv/bin:$PATH"
ENV XDG_CONFIG_HOME=/tmp/prophet-config \
    XDG_CACHE_HOME=/tmp/prophet-cache
CMD ["prophet", "worker"]
