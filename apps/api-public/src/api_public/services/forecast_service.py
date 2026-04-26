from pathlib import Path
import yaml
import torch

from forecast_core.data_pipelines.omni_dscovr_sdo.scalers import ForecastScalers
from forecast_core.inference.predict import ForecastInferenceService
from forecast_core.inference.request_builder import build_inference_batch
from forecast_core.inference.serialization import load_checkpoint
from forecast_core.models.forecast_model import ForecastModel
from forecast_core.solar_encoder.embedding_loader import EmbeddingStore


class ForecastService:
    def __init__(self, config_path: Path, checkpoint_path: Path, scaler_path: Path, embeddings_path: Path | None = None, embeddings_format: str = 'npz'):
        self.config = yaml.safe_load(config_path.read_text())
        self.scalers = ForecastScalers.from_yaml(scaler_path)
        self.model = ForecastModel(self.config)
        if checkpoint_path.exists():
            load_checkpoint(self.model, checkpoint_path)
        self.model.eval()
        self.embedding_store = None
        if embeddings_path and embeddings_path.exists():
            self.embedding_store = EmbeddingStore.from_npz(embeddings_path) if embeddings_format == 'npz' else EmbeddingStore.from_csv(embeddings_path)
        self.service = ForecastInferenceService(self.model, self.config, self.scalers.target)

    def predict(self, payload: dict) -> dict:
        with torch.no_grad():
            batch = build_inference_batch(
                payload=payload,
                history_scaler=self.scalers.history,
                context_scaler=self.scalers.context,
                target_scaler=self.scalers.target,
                expected_embed_dim=self.config['model']['embed_dim'],
                embedding_store=self.embedding_store,
                default_zero_embedding=True,
            )
            model_batch = {
                'image_embed': batch.image_embed,
                'image_embed_mask': batch.image_embed_mask,
                'history': batch.history,
                'context': batch.context,
                'persistence': batch.persistence,
            }
            return self.service.predict(model_batch, reference_timestamp=batch.reference_timestamp)
