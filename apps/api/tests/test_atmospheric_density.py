from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.atmospheric_density import router
from app.routers.forecasts import router as general_router
from app.services import atmospheric_density as service


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    issue = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    frame = pd.DataFrame([
        dict(issue_time=issue, observed_at=issue - timedelta(days=5),
             valid_time=issue + timedelta(hours=lead), lead_hours=lead,
             driver_mode='observed_persistence', background_method='trailing_81_daily_values',
             dtc_method='causal_dst_ap_v1', history_start=issue-timedelta(days=85),
             dtc_observed_at=issue, altitude_km=400, latitude_deg=0,
             rho_kg_m3=1e-12, rho_lon_p10_kg_m3=8e-13, rho_lon_p90_kg_m3=2e-12)
        for lead in range(49)
    ])
    path = tmp_path / 'density.csv'
    frame.to_csv(path, index=False)
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(
        workdir=tmp_path, models_registry={'models': {'atmospheric_density': {
            'max_age_hours': 6}}}))
    # Rendering tests receive a frame from the mocked Prophet boundary.
    monkeypatch.setattr(service, "read_frames", lambda _: {"atmospheric_density": pd.read_csv(path)})
    return path, frame


def client():
    app = FastAPI()
    app.include_router(router)
    app.include_router(general_router)
    return TestClient(app)


def test_get_returns_internal_grid_without_inputs(artifact):
    with client() as api:
        response = api.get('/public/forecasts/atmospheric-density')
        assert response.status_code == 200
        data = response.json()['data']
        assert data['driver_mode'] == 'observed_persistence'
        assert data['background_method'] == 'trailing_81_daily_values'
        assert data['dtc_method'] == 'causal_dst_ap_v1'
        assert len(data['predictions']) == 49
        assert data['predictions'][0]['cells'][0]['rho_kg_m3'] == 1e-12
        assert api.post('/public/forecasts/atmospheric-density', json={}).status_code == 405
        spec = api.get('/openapi.json').json()['paths']['/public/forecasts/atmospheric-density']
        assert 'post' not in spec
        assert 'requestBody' not in spec['get']


@pytest.mark.parametrize('failure', ['missing', 'stale', 'invalid', 'partial', 'wrong_mode'])
def test_unavailable_artifact_returns_503(artifact, failure):
    path, frame = artifact
    if failure == 'missing':
        path.unlink()
    else:
        if failure == 'stale':
            for name in ('issue_time', 'observed_at', 'valid_time', 'history_start', 'dtc_observed_at'):
                frame[name] -= timedelta(days=1)
        elif failure == 'invalid':
            frame.loc[0, 'rho_kg_m3'] = float('nan')
        elif failure == 'partial':
            frame = frame.iloc[:-1]
        else:
            frame['driver_mode'] = 'forecast_inputs'
        frame.to_csv(path, index=False)
    with client() as api:
        assert api.get('/public/forecasts/atmospheric-density').status_code == 503


def test_gap_filled_metadata_is_returned_and_required(artifact):
    import json
    path, frame = artifact
    frame['background_method'] = 'trailing_81_daily_values_linear_gapfill'
    dates = {'s10': ['2026-09-03', '2026-09-04']}
    frame['background_interpolated_days'] = json.dumps(dates)
    frame.to_csv(path, index=False)
    with client() as api:
        result = api.get('/public/forecasts/atmospheric-density')
        assert result.status_code == 200
        assert result.json()['data']['background_interpolated_days'] == dates
        frame.drop(columns='background_interpolated_days').to_csv(path, index=False)
        assert api.get('/public/forecasts/atmospheric-density').status_code == 503
