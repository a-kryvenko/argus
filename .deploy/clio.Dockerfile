FROM python:3.12-slim

WORKDIR /var/www/apps/clio

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

COPY apps/clio/pyproject.toml apps/clio/uv.lock ./

RUN uv sync --frozen --no-cache --no-install-local

COPY packages/common /var/www/packages/common

COPY apps/clio/src ./src

RUN uv sync --frozen --no-cache

ENV PATH="/var/www/apps/clio/.venv/bin:$PATH"

ENV XDG_CONFIG_HOME=/tmp/clio-config \
    XDG_CACHE_HOME=/tmp/clio-cache

RUN install -d -m 1777 "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME"

CMD ["clio", "serve"]
