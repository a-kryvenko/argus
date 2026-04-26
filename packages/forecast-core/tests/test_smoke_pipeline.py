from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve()
for candidate in [ROOT.parents[3] if 'forecast-core' in str(ROOT) else ROOT.parents[2]]:
    sys.path.insert(0, str(candidate / 'packages' / 'forecast-core' / 'src'))
    sys.path.insert(0, str(candidate / 'packages' / 'common' / 'src'))
    sys.path.insert(0, str(candidate / 'apps' / 'api-public' / 'src'))

import numpy as np
import pandas as pd
import torch
import yaml

from forecast_core.data_pipelines.omni_dscovr_sdo.compute_scalers import compute_scalers_from_dataset
from forecast_core.data_pipelines.omni_dscovr_sdo.prepare_dataset import DatasetBuilder
from forecast_core.data_pipelines.omni_dscovr_sdo.scalers import ForecastScalers
from forecast_core.models.forecast_model import ForecastModel
from forecast_core.solar_encoder.embedding_loader import EmbeddingStore


def _make_synthetic_df() -> pd.DataFrame:
    n = 220
    ts = pd.date_range('2025-01-01T00:00:00Z', periods=n, freq='h')
    return pd.DataFrame({
        'timestamp': ts,
        'V': 420 + 30 * np.sin(np.arange(n) / 12),
        'BX_GSE': 2 * np.sin(np.arange(n) / 8),
        'BY_GSM': 3 * np.cos(np.arange(n) / 10),
        'BZ_GSM': -2 + 4 * np.sin(np.arange(n) / 9),
        'N': 6 + 1.5 * np.cos(np.arange(n) / 7),
    })


def _write_scalers(dataset: dict[str, np.ndarray], path: Path) -> ForecastScalers:
    compute_scalers_from_dataset(dataset, path)
    return ForecastScalers.from_yaml(path)


def test_dataset_builder_smoke_shapes() -> None:
    df = _make_synthetic_df()
    builder = DatasetBuilder(history_hours=24, forecast_hours=12, embed_dim=1280)

    dataset = builder.build(df)

    expected_samples = len(df) - 24 - 12 + 1
    assert dataset['image_embed'].shape == (expected_samples, 1280)
    assert dataset['image_embed_mask'].shape == (expected_samples, 1)
    assert dataset['embedding_age_hours'].shape == (expected_samples, 1)
    assert dataset['history'].shape == (expected_samples, 24, 5)
    assert dataset['context'].shape[0] == expected_samples
    assert dataset['persistence'].shape == (expected_samples, 4)
    assert dataset['targets'].shape == (expected_samples, 12, 4)
    assert dataset['bz_event_targets'].shape == (expected_samples, 12)
    assert np.all(dataset['image_embed_mask'] == 0.0)


def test_dataset_builder_uses_embedding_store() -> None:
    df = _make_synthetic_df()
    timestamps = pd.to_datetime(df['timestamp'].iloc[10:210:4], utc=True)
    embeddings = np.random.default_rng(7).normal(size=(len(timestamps), 1280)).astype(np.float32)
    store = EmbeddingStore(timestamps, embeddings)

    builder = DatasetBuilder(history_hours=24, forecast_hours=12, embed_dim=1280, embedding_store=store, max_embedding_age_hours=12.0)
    dataset = builder.build(df)

    assert dataset['image_embed'].shape[1] == 1280
    assert np.any(dataset['image_embed_mask'] == 1.0)
    assert np.all(dataset['embedding_age_hours'][dataset['image_embed_mask'].squeeze(-1) == 1.0] >= 0.0)


def test_model_forward_smoke(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    builder = DatasetBuilder(history_hours=24, forecast_hours=12, embed_dim=1280)
    dataset = builder.build(df)

    scalers = _write_scalers(dataset, tmp_path / 'scalers.yaml')
    config = yaml.safe_load(Path('packages/forecast-core/configs/forecast/inference.yaml').read_text())
    config['model']['output_steps'] = 12

    model = ForecastModel(config)
    model.eval()

    image_embed = torch.tensor(dataset['image_embed'][:2], dtype=torch.float32)
    image_embed_mask = torch.tensor(dataset['image_embed_mask'][:2], dtype=torch.float32)
    history = scalers.history.normalize(torch.tensor(dataset['history'][:2], dtype=torch.float32))
    context = scalers.context.normalize(torch.tensor(dataset['context'][:2], dtype=torch.float32))
    persistence = scalers.target.normalize(torch.tensor(dataset['persistence'][:2], dtype=torch.float32))

    with torch.no_grad():
        outputs = model(
            image_embed=image_embed,
            image_embed_mask=image_embed_mask,
            history=history,
            context=context,
            persistence=persistence,
        )

    assert outputs['mean'].shape == (2, 12, 4)
    assert outputs['std'].shape == (2, 12, 4)
    assert outputs['bz_event_logits'].shape == (2, 12)
    assert outputs['regime_logits'].shape == (2, 4)
