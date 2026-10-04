FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git openssh-client \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /var/www/apps/clio
COPY packages/common /var/www/packages/common
COPY apps/clio/pyproject.toml apps/clio/uv.lock ./
COPY apps/clio/src ./src
RUN uv sync --frozen --no-cache
ENV PATH="/var/www/apps/clio/.venv/bin:$PATH"
ENV XDG_CONFIG_HOME=/tmp/clio-config \
    XDG_CACHE_HOME=/tmp/clio-cache
CMD ["clio", "serve"]
