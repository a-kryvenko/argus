from datetime import UTC, datetime, timedelta
import hashlib
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from common.schemas.forecast_release import ForecastRelease
from common.schemas.leo_drag import LeoDragRequest
from argus_intelligence import drag, main

REQUEST = dict(altitude_km=400, inclination_deg=51.6, mass_kg=100,
               effective_area_m2=1, drag_coefficient=2.2, horizon_hours=24)


@pytest.fixture
def release(monkeypatch):
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    frame = pd.DataFrame([
        dict(issue_time=now, observed_at=now-timedelta(days=1), valid_time=now+timedelta(hours=i),
             lead_hours=i, driver_mode='observed_persistence', background_method='trailing_81_daily_values',
             dtc_method='causal_dst_ap_v1', history_start=now-timedelta(days=85), dtc_observed_at=now,
             altitude_km=h, latitude_deg=lat, rho_kg_m3=1e-12,
             rho_lon_p10_kg_m3=0.8e-12, rho_lon_p90_kg_m3=1.2e-12)
        for i in range(49) for h in (200, 400, 800) for lat in (-90, 0, 90)])

    def build(frame):
        csv = frame.to_csv(index=False)
        return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='atmospheric-density',
                               issue_time=frame.issue_time.iloc[0], published_at=frame.issue_time.iloc[0],
                               artifacts=[dict(name='atmospheric_density', csv_text=csv,
                                               sha256=hashlib.sha256(csv.encode()).hexdigest(),
                                               row_count=len(frame), columns=list(frame.columns), model_info={})])
    monkeypatch.setattr(drag, 'fetch_release', lambda _: build(frame))
    monkeypatch.setattr(drag, 'get_config', lambda: SimpleNamespace(models_registry={'models': {'atmospheric_density': {'max_age_hours': 6}}}))
    monkeypatch.setenv('INTELLIGENCE_SERVICE_TOKEN', 'impact-token')
    return frame


@pytest.fixture
def backend(monkeypatch):
    module = ModuleType('intelligence_core.api')
    module.ModelDomainError = type('ModelDomainError', (ValueError,), {})
    def calculate(**kwargs):
        return dict(mean_density_kg_m3=1e-12, mean_drag_accel_m_s2=1e-6,
                    delta_v_loss_m_s=0.1, estimated_altitude_loss_m=100,
                    predictions=[dict(mean_density_kg_m3=1e-12, mean_drag_accel_m_s2=1e-6,
                                      mean_along_track_deceleration_m_s2=1e-6,
                                      delta_v_loss_m_s=i/(len(kwargs['grids'])-1)*0.1, estimated_altitude_loss_m=100*i/(len(kwargs['grids'])-1))
                                 for i in range(len(kwargs['grids']))])
    module.calculate_drag = calculate
    monkeypatch.setitem(__import__('sys').modules, 'intelligence_core.api', module)
    return module


@pytest.mark.parametrize(('thresholds', 'risk'), [
    (None, 'not_assessed'),
    (dict(elevated_altitude_loss_m=101, high_altitude_loss_m=200), 'low'),
    (dict(elevated_altitude_loss_m=100, high_altitude_loss_m=200), 'elevated'),
    (dict(elevated_altitude_loss_m=50, high_altitude_loss_m=100), 'high'),
])
def test_thresholds_and_provenance(release, backend, thresholds, risk):
    result = drag.assess_drag(LeoDragRequest(**REQUEST, thresholds=thresholds))
    assert result.drag_risk == risk
    assert result.start_time == release.issue_time.iloc[0]
    assert result.end_time - result.start_time == timedelta(hours=24)
    assert result.source.observed_at == release.observed_at.iloc[0]
    assert len(result.predictions) == 25
    assert 'p_elevated_drag' not in result.model_dump()


@pytest.mark.parametrize('failure', ['stale', 'future', 'wrong-mode', 'partial', 'hole', 'negative', 'duplicate', 'time'])
def test_invalid_density_is_unavailable(release, backend, failure):
    if failure in ('stale', 'future'):
        for name in ('issue_time', 'valid_time', 'observed_at', 'dtc_observed_at', 'history_start'):
            release[name] += timedelta(hours=-7 if failure == 'stale' else 2)
    elif failure == 'wrong-mode':
        release['driver_mode'] = 'forecast'
    elif failure == 'partial':
        release.drop(release.index[-1], inplace=True)
    elif failure == 'hole':
        release.drop(release[(release.altitude_km == 400) & (release.latitude_deg == 0)].index, inplace=True)
    elif failure == 'negative':
        release.loc[0, 'rho_kg_m3'] = -1
    elif failure == 'duplicate':
        release.loc[1] = release.loc[0]
    else:
        release.loc[0, 'valid_time'] += timedelta(hours=1)
    with TestClient(main.app) as client:
        response = client.post('/internal/v1/risks/leo-drag', json=REQUEST,
                               headers={'Authorization': 'Bearer impact-token'})
    assert response.status_code == 503


def test_auth_and_domain_errors(release, backend, monkeypatch):
    with TestClient(main.app) as client:
        assert client.post('/internal/v1/risks/leo-drag', json=REQUEST).status_code == 401
        assert client.post('/internal/v1/risks/leo-drag', json=REQUEST,
                           headers={'Authorization': 'Bearer wrong'}).status_code == 401
        def fail(**kwargs):
            raise backend.ModelDomainError('Outside grid')
        monkeypatch.setattr(backend, 'calculate_drag', fail)
        assert client.post('/internal/v1/risks/leo-drag', json=REQUEST,
                           headers={'Authorization': 'Bearer impact-token'}).status_code == 422
        monkeypatch.delenv('INTELLIGENCE_SERVICE_TOKEN')
        assert client.post('/internal/v1/risks/leo-drag', json=REQUEST).status_code == 503


def test_real_backend_end_to_end(release):
    pytest.importorskip('intelligence_core.api')
    with TestClient(main.app) as client:
        response = client.post('/internal/v1/risks/leo-drag', json={**REQUEST, 'horizon_hours': 48},
                               headers={'Authorization': 'Bearer impact-token'})
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result['predictions']) == 49
    assert 0 < result['estimated_altitude_loss_m'] < 1000
    assert result['predictions'][-1]['estimated_altitude_loss_m'] == result['estimated_altitude_loss_m']
