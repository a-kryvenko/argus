FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision==0.27.1
RUN uv pip install --system timm==1.0.27 scikit-learn==1.9.0 scikit-image==0.26.0 \
    'sunpy[map]==7.1.1' aiapy==0.11.0 astropy==8.0.0 drms==0.9.1 reproject==0.21.0 \
    numpy==2.4.6 'pandas>=2.2,<3' scipy httpx pydantic pydantic-settings pyyaml python-dotenv
WORKDIR /var/www
COPY packages/common/src /var/www/packages/common/src
COPY packages/forecast/src /var/www/packages/forecast/src
COPY apps/prophet/src /var/www/apps/prophet/src
ENV PYTHONPATH=/var/www/packages/common/src:/var/www/packages/forecast/src:/var/www/apps/prophet/src \
    XDG_CACHE_HOME=/tmp/proswin-cache TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor SUNPY_DOWNLOADDIR=/tmp/sunpy-downloads SUNPY_CONFIGDIR=/tmp/sunpy MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=4
CMD ["python", "-m", "argus_prophet.services.proswin_queue"]
