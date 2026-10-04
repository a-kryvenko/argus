FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /var/www/apps/intelligence
COPY apps/intelligence/pyproject.toml apps/intelligence/uv.lock ./

# Keep third-party dependencies cached when application or local package code changes.
RUN uv sync --frozen --no-cache --no-install-local

COPY packages/intelligence-core /var/www/packages/intelligence-core
COPY packages/common /var/www/packages/common
COPY apps/intelligence/src ./src
RUN uv sync --frozen --no-cache
ENV PATH="/var/www/apps/intelligence/.venv/bin:$PATH"
CMD ["intelligence", "check"]
