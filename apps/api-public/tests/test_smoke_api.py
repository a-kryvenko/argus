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
import yaml
from fastapi.testclient import TestClient

from api_public.main import app
from api_public.services.forecast_service import ForecastService
from forecast_core.data_pipelines.omni_dscovr_sdo.compute_scalers import compute_scalers_from_dataset
from forecast_core.data_pipelines.omni_dscovr_sdo.prepare_dataset import DatasetBuilder

REPO_ROOT = Path(__file__).resolve().parents[3]


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


def _write_scalers(dataset: dict, path: Path) -> None:
    compute_scalers_from_dataset(dataset, path)


def test_forecast_endpoint_smoke(tmp_path: Path, monkeypatch) -> None:
    df = _make_synthetic_df()
    builder = DatasetBuilder(history_hours=24, forecast_hours=12, embed_dim=1280)
    dataset = builder.build(df)

    config = yaml.safe_load((REPO_ROOT / 'packages' / 'forecast-core' / 'configs' / 'forecast' / 'inference.yaml').read_text())
    config['model']['output_steps'] = 12
    config_path = tmp_path / 'inference.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    scaler_path = tmp_path / 'scalers.yaml'
    _write_scalers(dataset, scaler_path)

    service = ForecastService(
        config_path=config_path,
        checkpoint_path=tmp_path / 'missing-checkpoint.pt',
        scaler_path=scaler_path,
        embeddings_path=None,
        embeddings_format='npz',
    )

    from api_public.routes import forecast as forecast_route_module

    monkeypatch.setattr(forecast_route_module, 'get_forecast_service', lambda: service)

    client = TestClient(app)
    history_df = df.tail(24).copy()
    history_df['timestamp'] = history_df['timestamp'].apply(lambda x: x.isoformat())
    payload = {
        'reference_timestamp': df.iloc[-1]['timestamp'].isoformat(),
        'history': history_df.to_dict(orient='records'),
    }

    response = client.post('/forecast', json=payload)

    assert response.status_code == 200
    body = response.json()
    assert 'points' in body
    assert 'regime_probabilities' in body
    assert len(body['points']) == 12
    first = body['points'][0]
    assert {'timestamp', 'bx', 'by', 'bz', 'density', 'std_bx', 'std_by', 'std_bz', 'std_density', 'southward_bz_probability'} <= set(first.keys())
