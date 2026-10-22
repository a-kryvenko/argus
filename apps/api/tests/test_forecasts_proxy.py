import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import httpx
import pytest

from common.schemas.forecast_release import ForecastRelease, ForecastArtifact
from api.services.forecasts import client as forecasts_client
from api.services.forecasts import products as forecast_products
from api.services.errors import ArtifactNotReadyError


def test_wind_speed_serializes_as_integers_without_rounding_probabilities():
    variable = forecast_products.Variable('v', 'plasma_speed_quantile', 'km/s', 'v', (450,), True)
    result = forecast_products._variable_forecast(variable, {
        'v_q10': 380.2, 'v_q50': 420.7, 'v_q90': 480.9, 'p_v_ge_450': 0.23456,
    })
    data = json.loads(result.model_dump_json())
    assert data['continuous'] == {'q10': 380, 'q50': 421, 'q90': 481}
    assert all(type(value) is int for value in data['continuous'].values())
    assert type(data['binary'][0]['threshold']) is int
    assert data['binary'][0]['probability'] == 0.23456


def test_other_forecast_variables_keep_fractional_values():
    variable = forecast_products.Variable('n', 'plasma_density_quantile', 'cm^-3', 'n', quantiles=True)
    result = forecast_products._variable_forecast(variable, {'n_q10': 1.2, 'n_q50': 3.7, 'n_q90': 8.9})
    assert json.loads(result.model_dump_json())['continuous'] == {'q10': 1.2, 'q50': 3.7, 'q90': 8.9}


def payload():
    content = 'issue_time,valid_time,lead_hours,dst_q10,dst_q50,dst_q90\n2026-09-13T00:00:00Z,2026-09-13T01:00:00Z,1,-20,-10,0\n'
    return ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='dst',
        issue_time=datetime(2026, 9, 13, tzinfo=UTC), published_at=datetime.now(UTC),
        artifacts=[ForecastArtifact(name='dst_quantile', csv_text=content,
            sha256=hashlib.sha256(content.encode()).hexdigest(), row_count=1,
            columns=content.splitlines()[0].split(','), model_info={})]).model_dump(mode='json')


def install_transport(monkeypatch, response):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet-api:8000')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'secret')
    real_client = httpx.Client
    def handle(request):
        assert request.url.path == '/internal/v1/forecasts/dst/latest'
        assert request.headers['Authorization'] == 'Bearer secret'
        return response
    monkeypatch.setattr(forecasts_client.httpx, 'Client', lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))


def test_public_forecast_renders_http_release_without_local_files(monkeypatch):
    install_transport(monkeypatch, httpx.Response(200, json=payload()))
    monkeypatch.setattr(forecast_products, 'get_config', lambda: pytest.fail('Forecast must not read local paths'))
    result = forecast_products.load_forecast(forecast_products.PRODUCTS['dst'])
    assert result.predictions[0].variables['dst'].continuous.q50 == -10


@pytest.mark.parametrize('failure', ['unavailable', 'corrupt', 'wrong_product', 'version', 'incomplete'])
def test_upstream_failures_do_not_fall_back_to_csv(monkeypatch, failure):
    data = payload()
    if failure == 'corrupt':
        data['artifacts'][0]['csv_text'] += 'modified'
    elif failure == 'wrong_product':
        data['product'] = 'hmf'
    elif failure == 'version':
        data['contract_version'] = 2
    elif failure == 'incomplete':
        data['artifacts'] = []
    install_transport(monkeypatch, httpx.Response(503 if failure == 'unavailable' else 200, json=data))
    with pytest.raises(ArtifactNotReadyError):
        forecasts_client.read_frames('dst')


def test_missing_token_fails_without_http(monkeypatch):
    monkeypatch.delenv('FORECASTS_SERVICE_TOKEN', raising=False)
    with pytest.raises(ArtifactNotReadyError, match='not configured'):
        forecasts_client.read_release('dst')


def test_hmf_preserves_southward_after_total_horizon(monkeypatch):
    import pandas as pd
    product = forecast_products.PRODUCTS['hmf']
    issue = pd.Timestamp('2026-09-24T00:00Z')
    frames = []
    for variable, horizon in zip(product.variables, [24, 48]):
        frame = pd.DataFrame({'lead_hours': range(1,horizon+1)})
        frame['issue_time'] = issue
        frame['valid_time'] = issue + pd.to_timedelta(frame.lead_hours, unit='h')
        for threshold in [5,10,15]:
            frame[f'p_{variable.name}_ge_{threshold}'] = .1
        frames.append((variable,frame))
    monkeypatch.setattr(forecast_products, '_load_variable_frames', lambda _:frames)
    result = forecast_products.load_forecast(product)
    assert result.horizon_hours == 48
    assert set(result.predictions[23].variables) == {'bt','bs'}
    assert set(result.predictions[24].variables) == {'bs'}


@pytest.mark.parametrize('failure', [None, 'unavailable', 'wrong_product', 'corrupt'])
def test_verification_proxy_validates_upstream(monkeypatch, failure):
    monkeypatch.setenv('FORECASTS_URL', 'http://prophet-api:8000')
    monkeypatch.setenv('FORECASTS_SERVICE_TOKEN', 'secret')
    data = {'product': 'hmf' if failure == 'wrong_product' else 'dst',
            'start': '2026-09-03T00:00:00Z', 'end': '2026-10-03T00:00:00Z', 'groups': []}
    if failure == 'corrupt':
        data.pop('start')
    real_client = httpx.Client
    def handle(request):
        assert request.url.path == '/internal/v1/forecasts/dst/verification'
        assert request.headers['Authorization'] == 'Bearer secret'
        return httpx.Response(503 if failure == 'unavailable' else 200, json=data)
    monkeypatch.setattr(forecasts_client.httpx, 'Client', lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    if failure:
        with pytest.raises(ArtifactNotReadyError):
            forecasts_client.read_verification('dst')
    else:
        assert forecasts_client.read_verification('dst').groups == []


def test_verification_route_rejects_old_prefix_and_preserves_upstream_failure(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers.forecasts import router
    from common.schemas.forecast_verification import ForecastVerification, VerificationGroup
    app = FastAPI()
    app.include_router(router)
    expected = ForecastVerification(product='dst', start=datetime(2026, 9, 3, tzinfo=UTC),
        end=datetime(2026, 10, 3, tzinfo=UTC), groups=[VerificationGroup(
            artifact='dst_quantile', model='dst-v2', model_sha256='a' * 64,
            last_issue_time=datetime(2026, 10, 2, tzinfo=UTC),
            releases=2, evaluated_at=datetime(2026, 10, 3, tzinfo=UTC), counts={'verified': 12})])
    calls = []
    def read(product):
        calls.append(product)
        return expected
    monkeypatch.setattr(forecasts_client, 'read_verification', read)
    with TestClient(app) as client:
        assert client.get('/private/forecasts/dst/verification').status_code == 404
        assert calls == []
        response = client.get('/forecasts/dst/verification')
        assert response.status_code == 200
        group = response.json()['data']['groups'][0]
        assert group['model'] == 'dst-v2'
        assert group['model_sha256'] == 'a' * 64
        assert group['last_issue_time'] == '2026-10-02T00:00:00Z'
        assert group['counts']['verified'] == 12
        assert calls == ['dst']
        def unavailable(product):
            raise ArtifactNotReadyError('unavailable')
        monkeypatch.setattr(forecasts_client, 'read_verification', unavailable)
        assert client.get('/forecasts/dst/verification').status_code == 503


def test_temperature_public_forecast_contract(monkeypatch):
    import pandas as pd
    issue = pd.Timestamp('2026-10-09T12:00Z')
    frame = pd.DataFrame({'issue_time': [issue], 'valid_time': [issue+pd.Timedelta(hours=96)],
                          'lead_hours': [96], 't_q10': [10000.], 't_q50': [100000.], 't_q90': [200000.]})
    monkeypatch.setattr(forecast_products, 'read_frames', lambda _: {'plasma_temperature_quantile': frame})
    result = forecast_products.load_forecast(forecast_products.PRODUCTS['solar-wind-temperature'], meta=True)
    assert result.horizon_hours == 96 and result.available_variables == ['t']
    assert result.meta.variables['t'].unit == 'K'
    assert result.predictions[0].variables['t'].continuous.q50 == 100000.
    content = frame.to_csv(index=False)
    ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='solar-wind-temperature',
        issue_time=issue, published_at=datetime.now(UTC), artifacts=[ForecastArtifact(
            name='plasma_temperature_quantile', csv_text=content, columns=list(frame), row_count=1,
            sha256=hashlib.sha256(content.encode()).hexdigest(), model_info={})])


@pytest.mark.parametrize('version', ['solar-wind-events-v1', 'solar-wind-events-v2', 'solar-wind-events-v3', 'solar-wind-events-v4'])
def test_events_route_reads_typed_report_and_preserves_source_availability(monkeypatch, version):
    import csv
    import io
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers.forecasts import router
    from common.schemas.solar_wind_events import SolarWindEventForecast, HSSForecast, CMEEnsembleForecast
    issue = datetime(2026, 10, 13, 12, 5, tzinfo=UTC)
    hss = [] if version.endswith('v1') else [HSSForecast(forecast_id='hss1', status='expected',
        timing_method='proswin_profile', arrival_at=issue, arrival_window_start=issue,
        arrival_window_end=issue, window_kind='uncalibrated_timing_envelope', evidence_sources=['PROSWIN'])]
    cmes = [] if version != 'solar-wind-events-v3' else [CMEEnsembleForecast(
        forecast_id='dbem:cme1', cme_id='cme1', source_poll_id=uuid4(), source_sha256='a'*64,
        analysis_sha256='b'*64, available_at=issue, initial_time=issue, initial_parameters={}, policy={},
        seed=42, ensemble_size=4096, geometric_hit_count=0, geometric_hit_fraction=0,
        arrival_count=0, right_censored_count=0, future_arrival_count=0,
        arrival_within_horizon_count=0, status='geometric_miss')]
    report = SolarWindEventForecast(issue_time=issue, events=[], l1_detections=[], model_version=version, hss_forecasts=hss,
                                   cme_ensembles=cmes,
                                   source_status={'CME': {'state': 'unavailable'}})
    if version == 'solar-wind-events-v4':
        from common.schemas.solar_wind_events import CMESpeedOutlook
        report.cme_speed = CMESpeedOutlook(model_version='cme-speed-v2',state='available',points=[dict(
            valid_time=issue.replace(hour=13),speed_background=400,speed_nominal=800,
            speed_early=600,speed_late=400,scenario_lower=400,scenario_upper=1000)])
    columns = ['issue_time', 'valid_time', 'lead_hours', 'event_count', 'detection_count', 'report_json']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerow(dict(issue_time=issue.isoformat(), valid_time=issue.isoformat(), lead_hours=0,
                         event_count=0, detection_count=0, report_json=report.model_dump_json()))
    content = stream.getvalue()
    release = ForecastRelease(product='solar-wind-events', release_id=uuid4(), run_id=uuid4(),
        issue_time=issue, published_at=issue, artifacts=[ForecastArtifact(
            name='solar_wind_events', csv_text=content, sha256=hashlib.sha256(content.encode()).hexdigest(),
            columns=columns, row_count=1, model_info={})])
    monkeypatch.setattr(forecasts_client, 'read_release', lambda product: release)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        response = client.get('/forecasts/solar-wind-events')
        assert response.status_code == 200
        assert response.json()['data']['experimental'] is True
        assert response.json()['data']['source_status']['CME']['state'] == 'unavailable'
        assert response.json()['data']['model_version'] == version
        assert len(response.json()['data']['hss_forecasts']) == len(hss)
        assert len(response.json()['data']['cme_ensembles']) == len(cmes)
        if version == 'solar-wind-events-v4':
            assert response.json()['data']['cme_speed']['points'][0]['speed_nominal'] == 800
            assert response.json()['data']['cme_speed']['scenario_kind'] == 'conditional_scenarios_not_quantiles'
        if cmes:
            assert response.json()['data']['cme_ensembles'][0]['calibrated'] is False
            assert response.json()['data']['cme_ensembles'][0]['source_poll_id'] == str(cmes[0].source_poll_id)
        def unavailable(product):
            raise ArtifactNotReadyError('unavailable')
        monkeypatch.setattr(forecasts_client, 'read_release', unavailable)
        assert client.get('/forecasts/solar-wind-events').status_code == 503


def test_speed_serves_CME_from_same_artifact_without_reading_event_release(monkeypatch):
    import pandas as pd
    from common.schemas.solar_wind_events import CMESpeedOutlook
    issue=pd.Timestamp('2026-10-14T12:00Z')
    valid=issue+pd.offsets.Hour(1)
    speed=CMESpeedOutlook(model_version='cme-speed-v2',state='available',points=[dict(
        valid_time=valid,speed_background=400,speed_nominal=800,speed_early=600,speed_late=400,
        scenario_lower=400,scenario_upper=1000)])
    quantiles=pd.DataFrame([dict(issue_time=issue,valid_time=valid,lead_hours=1,v_q10=350,v_q50=400,v_q90=450,
        cme_speed_json=speed.model_dump_json())])
    thresholds=pd.DataFrame([dict(issue_time=issue,valid_time=valid,lead_hours=1,
        p_v_ge_450=.1,p_v_ge_500=.05,p_v_ge_600=.01)])
    monkeypatch.setattr(forecast_products,'read_frames',lambda product:{'plasma_speed_quantile':quantiles,'plasma_speed_threshold':thresholds})
    monkeypatch.setattr(forecasts_client,'read_event_forecast',lambda:pytest.fail('Must not mix releases'))
    result=forecast_products.load_forecast(forecast_products.PRODUCTS['solar-wind-speed'])
    assert result.predictions[0].variables['v'].continuous.q50==400
    assert result.predictions[0].variables['v'].binary[0].probability==.1
    assert result.cme_speed.points[0].speed_nominal==800
    assert result.cme_speed.scenario_kind=='conditional_scenarios_not_quantiles'
    quantiles.drop(columns=['cme_speed_json'],inplace=True)
    assert forecast_products.load_forecast(forecast_products.PRODUCTS['solar-wind-speed']).cme_speed is None
