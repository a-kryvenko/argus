from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    artifact_dir: Path = Path("data/artifacts/forecast")
    model_path: Path = Path("data/artifacts/forecast/forecast_model.pt")
    config_path: Path = Path("packages/forecast-core/configs/forecast/inference.yaml")
    scaler_path: Path = Path("data/artifacts/forecast/scalers.yaml")
    embeddings_path: Path | None = None
    embeddings_format: str = "npz"


settings = Settings()
