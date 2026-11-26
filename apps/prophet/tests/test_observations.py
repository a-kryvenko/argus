import asyncio
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
    client = httpx.Client
    monkeypatch.setattr(observations.httpx, 'Client', lambda **kwargs: client(
        transport=httpx.MockTransport(handler), **kwargs))


def load_inputs(product='dst', cache=None):
    def execute(target, *args, timeout_seconds, **kwargs):
        return target(*args, **kwargs)

    return asyncio.run(observations.load_product_inputs(
        product, NOW, {} if cache is None else cache, executor=execute, timeout_seconds=5,
    ))


def resource_response(request, payload):
    resource = request.url.path.removeprefix('/internal/v1/observations/') if request.url.path != '/internal/v1/files' else 'files'
    if resource == 'model-inputs':
        assert 'interval' not in request.url.params
        data = {'interval': payload.get('interval', '1h'), 'points': [
            dict(time=point['issue_time'], **{k: v for k, v in point.items() if k != 'issue_time'})
            for point in payload['observations']['points']]}
    elif resource == 'series':
        data = {}
        start = NOW - timedelta(hours=1)
        for metric in request.url.params['metrics'].split(','):
            point = {'observed_at': start.isoformat(), 'value': payload['observations']['points'][0][metric],
                     'expected_count': 60, 'count': 59}
            if metric != 'bx':
                point['received_at'] = (NOW - timedelta(minutes=2)).isoformat()
            data[metric] = {'points': [point], 'time_basis': 'l1', 'aggregation': 'mean', 'resolution_seconds': 3600}
            if metric == 'f10_7':
                point['source_product'] = 'swpc.f107'
                noon = NOW.replace(hour=20) - timedelta(days=1)
                point['observed_at'] = noon.isoformat()
                data[metric].update(time_basis='measurement', aggregation='source', resolution_seconds=86400)
                data[metric]['points'].append({**point, 'observed_at': (noon+timedelta(hours=2)).isoformat(),
                                               'value': 999.})
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
    shared = {}
    assert load_inputs(cache=shared).observations.points[0].v == 420
    assert len(calls) == 1
    assert all('as_of' not in request.url.params for request in calls)
    assert calls[0].url.params['from'] == (NOW-timedelta(days=60)).isoformat()
    assert calls[0].url.path.endswith('/model-inputs')
    assert set(calls[0].url.params['metrics'].split(',')) >= {'f10_7', 'dst', 'ap', 's10_81mean'}
    native = load_inputs('hmf', shared).solar_wind_hourly
    assert calls[1].url.path.endswith('/series')
    assert calls[1].url.params['interval'] == '1h'
    assert calls[1].url.params['time_basis'] == 'l1'
    assert calls[1].url.params['from'] == (NOW-timedelta(hours=168)).isoformat()
    assert native['series']['bx']['points'] == []
    point = native['series']['bz']['points'][0]
    assert point['received_at'] == (NOW - timedelta(minutes=2)).isoformat()
    assert point['interval_end'] == NOW and point['count'] == 59
    radiation = load_inputs('solar-radiation', shared)
    assert radiation.as_of == NOW
    assert calls[2].url.params['observation'] == 'goes'
    assert calls[2].url.params['start'] == (NOW-timedelta(days=35)).isoformat()
    assert calls[3].url.params['received_by'] == NOW.isoformat()
    assert calls[3].url.params['metrics'] == 'f10_7'
    assert len(radiation.f107_measurements) == 1
    assert radiation.f107_measurements[0].received_at == NOW - timedelta(minutes=2)
    assert len(calls) == 4


def test_density_forecast_uses_only_available_closed_l1_history(monkeypatch):
    import io
    from pathlib import Path
    import yaml
    from prophet.services.generation.calculation import CalculationRequest, calculate_product

    calls = []

    def handler(request):
        calls.append(request.url.path)
        response = resource_response(request, stored_inputs().model_dump(mode='json'))
        payload = response.json()
        period = payload['data']['n']['points'][0]
        period['value'] = 2.0
        payload['data']['n']['points'].extend([
            {**period, 'observed_at': (NOW-timedelta(hours=2)).isoformat(),
             'received_at': (NOW+timedelta(seconds=1)).isoformat(), 'value': 45.0},
            {**period, 'observed_at': NOW.isoformat(), 'value': 60.0},
        ])
        return httpx.Response(200, json=payload)

    configure(monkeypatch, handler)
    inputs = load_inputs('solar-wind-density')
    assert calls == ['/internal/v1/observations/series']
    workdir = Path(__file__).resolve().parents[3]
    registry = yaml.safe_load((workdir / 'configs/models_registry.yaml').read_text())['models']
    artifact, = calculate_product(CalculationRequest('solar-wind-density', inputs, workdir, registry))
    frame = pd.read_csv(io.BytesIO(artifact.content))
    assert artifact.model_info['input_time_basis'] == 'l1'
    assert artifact.model_info['usable_hours'] == 1
    assert frame.lead_hours.tolist() == list(range(1, 121))
    assert frame.n_q50.tolist() == pytest.approx([2.0] * 120)
    assert frame.n_q10.between(0, frame.n_q50).all()
    assert frame.n_q90.ge(frame.n_q50).all()


def test_goes_failure_does_not_prevent_numeric_forecast(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith('/files'):
            return httpx.Response(503)
        return resource_response(request, stored_inputs().model_dump(mode='json'))

    configure(monkeypatch, handler)
    cache = {}
    for product in ('solar-radiation', 'atmospheric-density'):
        with pytest.raises(RuntimeError, match='goes input unavailable'):
            load_inputs(product, cache)
    assert load_inputs('dst', cache).observations.points[0].v == 420
    assert [request.url.path for request in calls] == ['/internal/v1/observations/model-inputs', '/internal/v1/files']



@pytest.mark.parametrize('status', [401, 503])
def test_owner_failure_is_propagated_without_fallback(monkeypatch, status):
    configure(monkeypatch, lambda _: httpx.Response(status))
    with pytest.raises(RuntimeError, match='hourly input unavailable') as error:
        load_inputs()
    assert isinstance(error.value.__cause__, httpx.HTTPStatusError)


def test_empty_inputs_fail_without_ingestion(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['observations']['points'] = []
    configure(monkeypatch, lambda request: resource_response(request, payload))
    with pytest.raises(RuntimeError, match='check ingestion'):
        load_inputs()


def test_rejects_incompatible_contract(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['interval'] = '5m'
    configure(monkeypatch, lambda request: resource_response(request, payload))
    with pytest.raises(RuntimeError, match='hourly input unavailable') as error:
        load_inputs()
    assert isinstance(error.value.__cause__, ValueError)


def test_fractional_inputs_are_converted_only_for_legacy_model_schema(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['observations']['points'][0]['f10_7'] = 112.75
    configure(monkeypatch, lambda request: resource_response(request, payload))
    result = load_inputs()
    assert result.observations.points[0].f10_7 == 112
    assert next(row.value for row in result.measurements if row.metric == 'f10_7') == 112.75


def test_verification_reads_compact_live_metrics_and_excludes_end(monkeypatch):
    from prophet.services.verification import read_targets
    times = pd.date_range('2026-01-01', periods=201, freq='min', tz='UTC')
    calls = []
    def handler(request):
        calls.append(request)
        assert request.url.path == '/internal/v1/observations/history'
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
    load = Mock(side_effect=lambda service, **_: (
        SimpleNamespace(registry_name=service.registry_name, last_status={}, model_info={}), {'sha256': 'test'},
    ))
    compute = Mock(side_effect=lambda service, *args, **kwargs:
                   ForecastResult(service.registry_name, pd.DataFrame({'value': [1]}), {}))
    from prophet.services.generation import models
    from forecast import calculation
    monkeypatch.setattr(models, 'load_model', load)
    monkeypatch.setattr(calculation, 'calculate_snapshot', compute)
    return compute


@pytest.mark.parametrize('selection,artifacts', [
    ('solar-wind-speed', ['plasma_speed_quantile', 'plasma_speed_threshold']),
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


def test_temperature_verification_uses_hourly_observations():
    from prophet.services.verification import targets
    frame = pd.DataFrame({'metric': ['t', 't'], 'observed_at': [NOW, NOW+timedelta(minutes=30)],
                          'value': [100000., 200000.]})
    assert targets(frame, NOW, NOW+timedelta(hours=1)) == [
        {'valid_time': NOW.isoformat(), 'metric': 't', 'value': 150000., 'sample_count': 2}]


def test_runtime_catalog_selects_explicit_services():
    from prophet.services.generation.products import PRODUCTS
    from forecast.inference.quantile import QuantileFS
    from forecast.inference.probability import ProbabilityFS
    for product, spec in PRODUCTS.items():
        if product in {'solar-wind-events', 'atmospheric-density'}:
            continue
        for model in spec.models:
            service = model.service_class()
            assert service.registry_name == model.artifact
            if model.outputs:
                assert callable(service.forecast_snapshot)
            else:
                assert issubclass(service, (QuantileFS, ProbabilityFS))
            assert service.__module__.startswith('forecast_core.inference.')


def test_assembles_hourly_measurements_and_goes_references(monkeypatch):
    def handler(request):
        if request.url.path.endswith('/model-inputs'):
            payload = stored_inputs().model_dump(mode='json')
            payload['observations']['points'][0].update(v=430., n=4.5, dst=-20.)
            return resource_response(request, payload)
        if request.url.path.endswith('/files') and request.url.params['observation'] == 'goes':
            return httpx.Response(200, json={'success': True, 'data': [{
                'observation': 'goes', 'observed_at': NOW.isoformat(),
                'sha256': 'a'*64, 'source_product': 'goes.live',
                'file': {'id':'00000000-0000-0000-0000-000000000001','timestamp':NOW.isoformat(),'bucket':'argus-private','key':'observations/goes/original.json'}}]})
        return resource_response(request, stored_inputs().model_dump(mode='json'))
    from prophet.services.inputs import solar_indices
    monkeypatch.setattr(solar_indices, 'original', lambda _: b'[{}]')
    monkeypatch.setattr(solar_indices, 'solar_measurements', lambda *_: [])
    configure(monkeypatch, handler)
    result = load_inputs('atmospheric-density')
    assert result.as_of == NOW
    assert next(row for row in result.measurements if row.metric == 'dst').value == -20.
    assert result.speed_observations[0].v == 430.
    assert result.density_observations[0].n == 4.5
    assert result.files[0].observation == 'goes'
