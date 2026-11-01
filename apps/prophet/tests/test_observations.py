from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import pytest
import pandas as pd
from forecast.calculation import ForecastResult
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation, ObservationPoint
from prophet.services.inputs import loader as observations

NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def stored_inputs():
    point = ObservationPoint(issue_time=NOW, bx=1, by=2, bz=-3, v=420, n=5,
                             t=100000, kp=2, dst=-10, ap=5, f10_7=120)
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[point]))


def configure(monkeypatch, handler):
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://observations')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(observations.httpx, 'Client', lambda **kwargs: client)


def resource_response(request, payload):
    resource = request.url.path.removeprefix('/internal/v1/observations/')
    if resource == 'metrics':
        interval = request.url.params['interval']
        data = {'interval': payload.get('interval', interval), 'points': [
            dict(time=point['issue_time'], **{k: v for k, v in point.items() if k != 'issue_time'})
            for point in payload['observations']['points']]}
    elif resource == 'files':
        data = []
    else:
        raise AssertionError(resource)
    return httpx.Response(200, json={'success': True, 'data': data})


def test_reads_generic_resources_and_owns_history_windows(monkeypatch):
    calls = []
    def handler(request):
        calls.append(request)
        assert request.headers['Authorization'] == 'Bearer test-token'
        return resource_response(request, stored_inputs().model_dump(mode='json'))
    configure(monkeypatch, handler)
    assert observations.load_inputs(NOW).observations.points[0].v == 420
    assert len(calls) == 5
    assert calls[0].url.params['from'] == (NOW-timedelta(days=60)).isoformat()
    assert calls[0].url.params['interval'] == '1h'
    assert set(calls[0].url.params['metrics'].split(',')) >= {'f10_7', 'dst', 'ap', 's10_81mean'}
    assert calls[1].url.params['interval'] == '1m'
    assert calls[1].url.params['from'] == (NOW-timedelta(hours=168)).isoformat()
    assert [request.url.params['kind'] for request in calls[2:]] == ['gong', 'goes', 'aia']
    assert calls[3].url.params['start'] == (NOW-timedelta(days=35)).isoformat()



@pytest.mark.parametrize('status', [401, 503])
def test_owner_failure_is_propagated_without_fallback(monkeypatch, status):
    configure(monkeypatch, lambda _: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        observations.load_inputs(NOW)


def test_empty_inputs_fail_without_ingestion(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['observations']['points'] = []
    configure(monkeypatch, lambda request: resource_response(request, payload))
    with pytest.raises(RuntimeError, match='check ingestion'):
        observations.load_inputs(NOW)


def test_rejects_incompatible_contract(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['interval'] = '5m'
    configure(monkeypatch, lambda request: resource_response(request, payload))
    with pytest.raises(ValueError):
        observations.load_inputs(NOW)


def test_verification_reads_compact_live_metrics_and_excludes_end(monkeypatch):
    from prophet.services.verification import read_targets
    times = pd.date_range('2026-01-01', periods=201, freq='min', tz='UTC')
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url.path == '/internal/v1/observations/metrics'
        assert request.headers['Authorization'] == 'Bearer test-token'
        assert request.url.params['interval'] == '1m'
        assert request.url.params['from'] == times[0].isoformat()
        assert request.url.params['to'] == times[-1].isoformat()
        return httpx.Response(200, json={'success': True, 'data': {'interval': '1m', 'points': [
            dict(time=time.isoformat(), v=400.) for time in times]}})
    configure(monkeypatch, handler)
    result = read_targets(times[0], times[-1])
    assert len(calls) == 1
    assert sum(row['sample_count'] for row in result['points']) == 200
    assert all(row['value'] == 400. for row in result['points'])


@pytest.mark.parametrize('failure', ['http', 'interval'])
def test_verification_rejects_failed_or_incompatible_reads(monkeypatch, failure):
    from prophet.services.verification import read_targets
    def handler(request):
        return httpx.Response(503) if failure == 'http' else httpx.Response(200, json={
            'success': True, 'data': {'interval': '1h', 'points': []}})
    configure(monkeypatch, handler)
    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        read_targets(NOW-timedelta(days=1), NOW)


def test_verification_empty_observations(monkeypatch):
    from prophet.services.verification import read_targets
    configure(monkeypatch, lambda _: httpx.Response(200, json={
        'success': True, 'data': {'interval': '1m', 'points': []}}))
    assert read_targets(NOW-timedelta(days=1), NOW)['points'] == []


def test_verification_targets_preserve_gaps_and_derive_before_averaging():
    from prophet.services.verification import targets
    frame = pd.DataFrame([
        {'observed_at': f'2026-01-01T{hour:02d}:{minute:02d}:00Z', 'metric': metric, 'value': value}
        for hour, minute, values in [(0, 0, {'bx': 3, 'by': 4, 'bz': -12, 'v': 400}),
                                     (0, 30, {'bx': -3, 'by': -4, 'bz': 12, 'v': 500}),
                                     (2, 0, {'v': 600}), (3, 0, {'v': 1000})]
        for metric, value in values.items()])
    result = targets(frame, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 3, tzinfo=UTC))
    assert [row['value'] for row in result if row['metric'] == 'v'] == [450., 600.]
    assert next(row for row in result if row['metric'] == 'bt')['value'] == 13
    assert next(row for row in result if row['metric'] == 'bs')['value'] == 6


def test_verification_incomplete_vectors_have_no_magnitude():
    from prophet.services.verification import targets
    frame = pd.DataFrame({'observed_at': ['2026-01-01T00:00:00Z'] * 2,
                          'metric': ['bx', 'by'], 'value': [3., 4.]})
    assert targets(frame, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)) == []


def mock_models(monkeypatch):
    from types import SimpleNamespace
    load = Mock(side_effect=lambda service, **_: (SimpleNamespace(registry_name=service.registry_name), {}))
    compute = Mock(side_effect=lambda service, *args, **kwargs:
                   ForecastResult(service.registry_name, pd.DataFrame({'value': [1]}), {}))
    from prophet.services.generation import models
    from forecast import calculation
    monkeypatch.setattr(models, 'load_model', load)
    monkeypatch.setattr(calculation, 'calculate_snapshot', compute)
    return compute


@pytest.mark.parametrize('selection,artifacts', [
    ('solar-wind-speed', ['plasma_speed_quantile', 'plasma_speed_threshold']),
    ('geomagnetic-activity', ['kp_threshold', 'ap_quantile']),
    ('hmf', ['hmf_total_threshold', 'hmf_southward_threshold']),
    ('dst', ['dst_quantile']),
    ('solar-wind-density', ['plasma_density_quantile']),
    ('solar-wind-temperature', ['plasma_temperature_quantile']),
    ('solar-radiation', ['f10_7_quantile', 's10_quantile', 'm10_quantile', 'y10_quantile']),
])
def test_selected_product_uses_one_snapshot(monkeypatch, selection, artifacts):
    from prophet.services.generation import calculation as generation
    inputs = stored_inputs()
    from pathlib import Path
    compute = mock_models(monkeypatch)
    results = generation.calculate_product(generation.CalculationRequest(selection, inputs, Path('/unused'), {}))
    assert [call.args[0].registry_name for call in compute.call_args_list] == artifacts
    for call in compute.call_args_list:
        assert call.args[1] is inputs
        assert call.kwargs['issue_time'] == NOW
    assert [result.name for result in results] == artifacts


def test_catalog_matches_public_contract():
    from prophet.services.generation.products import PRODUCTS
    from common.schemas.forecast_release import PRODUCT_ARTIFACTS
    assert set(PRODUCTS) == set(PRODUCT_ARTIFACTS)
    for name, product in PRODUCTS.items():
        assert product.artifacts == PRODUCT_ARTIFACTS[name]



def test_status_selections_include_canonical_names_and_historical_aliases():
    from prophet.services.generation.products import attempt_selections
    assert attempt_selections('solar-wind-speed') == ['all', 'solar-wind-speed', 'wind']
    assert attempt_selections('dst') == ['all', 'dst']
    assert attempt_selections('solar-radiation') == ['all', 'solar-radiation']


def test_temperature_verification_uses_hourly_observations():
    from prophet.services.verification import targets
    frame = pd.DataFrame({'metric': ['t', 't'], 'observed_at': [NOW, NOW+timedelta(minutes=30)],
                          'value': [100000., 200000.]})
    assert targets(frame, NOW, NOW+timedelta(hours=1)) == [
        {'valid_time': NOW.isoformat(), 'metric': 't', 'value': 150000., 'sample_count': 2}]


def test_runtime_catalog_selects_explicit_services():
    from prophet.services.generation.model_services import SERVICES
    from prophet.services.generation.products import PRODUCTS
    from forecast.inference.quantile import QuantileFS
    from forecast.inference.probability import ProbabilityFS
    for product, spec in PRODUCTS.items():
        if product in {'solar-wind-events', 'atmospheric-density'}:
            continue
        for model in spec.models:
            service = SERVICES[model.artifact]
            assert service.registry_name == model.artifact
            assert issubclass(service, (QuantileFS, ProbabilityFS))
            assert service.__module__.startswith('forecast_core.inference.')


def test_assembles_native_measurements_and_originals(monkeypatch):
    from prophet.services.inputs.observations import read_observations
    def handler(request):
        if request.url.path.endswith('/metrics'):
            payload = stored_inputs().model_dump(mode='json')
            payload['observations']['points'][0].update(v=430., n=4.5, dst=-20.)
            return resource_response(request, payload)
        if request.url.path.endswith('/files') and request.url.params['kind'] == 'gong':
            return httpx.Response(200, json={'success': True, 'data': [{
                'kind': 'gong', 'slot_at': NOW.isoformat(), 'observed_at': NOW.isoformat(),
                'available_at': NOW.isoformat(), 'sha256': 'a'*64, 'source_product': 'gong.live'}]})
        return resource_response(request, stored_inputs().model_dump(mode='json'))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = read_observations(client, 'http://observations', 'test-token', NOW)
    assert result.as_of == NOW
    assert next(row for row in result.measurements if row.metric == 'dst').value == -20.
    assert result.speed_observations[0].v == 430.
    assert result.density_observations[0].n == 4.5
    assert result.files[0].kind == 'gong'
