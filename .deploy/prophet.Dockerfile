FROM python:3.12-slim

WORKDIR /var/www/apps/prophet

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/prophet/pyproject.toml apps/prophet/uv.lock ./

RUN uv sync --frozen --no-cache --no-install-local

COPY packages/common /var/www/packages/common
COPY packages/forecast /var/www/packages/forecast
COPY packages/forecast-core /var/www/packages/forecast-core

COPY apps/prophet/src ./src

RUN uv sync --frozen --no-cache

ENV PATH="/var/www/apps/prophet/.venv/bin:$PATH"

ENV XDG_CONFIG_HOME=/tmp/prophet-config \
    XDG_CACHE_HOME=/tmp/prophet-cache

RUN install -d -m 1777 "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

CMD ["prophet", "serve"]
