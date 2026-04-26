FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
RUN apt update && apt install -y python3.11 python3-pip
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]