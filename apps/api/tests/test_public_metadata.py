"""Exercise HTTP serialization: compact and expanded responses share all data."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.routers.public import solar_wind, geomagnetic, observation_summary
from app.routers import forecasts
from app.services import forecast_products

TIME = '2026-09-30T00:00:00Z'
END = '2026-09-30T01:00:00Z'
WIND_META = dict(label='Bz', unit='nT', coordinate_system='GSM', source='NOAA SWPC RTSW',
                 source_url='https://example.test/wind', location='L1', time_basis='measurement',
                 propagated=False, resolution_seconds=60, aggregation='source_1_minute',
                 selection='NOAA active spacecraft', stale_after_seconds=600)
INDEX_META = dict(label='Estimated Kp', unit='', source='NOAA SWPC',
                  source_url='https://example.test/kp', resolution_seconds=10800, poll_seconds=60,
                  data_status='estimated', stale_after_seconds=14400, freshness_basis='interval_end',
                  time_basis='source_interval_start', gap_filling='none')
SAMPLE = dict(observed_at=TIME, received_at=TIME, value=None, spacecraft='DSCOVR',
              quality='flagged', provider_quality=1)
INDEX_SAMPLE = dict(interval_start=TIME, interval_end=END, interval_status='in_progress',
                    value=3.33, quality='unverified', received_at=TIME, station_count=None)
COVERAGE = dict(expected_slots=60, usable_slots=0, missing_slots=59, invalid_slots=1,
                percent=0, resolution_seconds=60, evaluated_to=END, basis='sample_timestamps',
                gaps=[{'from': TIME, 'to': END, 'reason': 'missing', 'slots': 59}])


def observation_payload(kind):
    wind = {**WIND_META, 'latest': SAMPLE, 'age_seconds': 900, 'status': 'stale'}
    index = {**INDEX_META, 'latest': INDEX_SAMPLE, 'lag_seconds': 0, 'status': 'fresh'}
    if kind == 'summary':
        return dict(generated_at=TIME, solar_wind={'bz': wind}, geomagnetic={'kp': index},
                    changes_1h={'bz': dict(status='unavailable', value=None, reason='stale', unit='nT')},
                    southward_bz=dict(status='lower_bound', value=12, as_of=TIME, unit='sampled_minutes',
                                      reason='onset_before_available_window'), lookback_minutes=75)
    if kind.endswith('latest'):
        return dict(generated_at=TIME, series={'bz': wind} if kind.startswith('solar') else {'kp': index})
    if kind.startswith('geomagnetic'):
        series = {'kp': {**INDEX_META, 'points': [INDEX_SAMPLE], 'coverage': {
            **COVERAGE, 'basis': 'overlapping_intervals', 'resolution_seconds': 10800}}}
    else:
        series = {'bz': {**WIND_META, 'points': [SAMPLE], 'coverage': COVERAGE}}
    return {'from': TIME, 'to': END, 'series': series, 'selection': 'native intervals', 'gap_filling': 'none'}


@pytest.mark.parametrize('module,kind', [
    (solar_wind, 'solar-wind/latest'), (solar_wind, 'solar-wind/history'),
    (geomagnetic, 'geomagnetic/latest'), (geomagnetic, 'geomagnetic/history'),
    (observation_summary, 'summary'),
])
def test_observation_metadata_is_opt_in_and_data_does_not_change(monkeypatch, module, kind):
    payload = observation_payload(kind)
    original = deepcopy(payload)
    read = AsyncMock(return_value={'success': True, 'data': payload, 'error': None})
    monkeypatch.setattr(module, 'read_observations', read)
    app = FastAPI()
    app.include_router(module.router)
    path = '/public/observations/' + kind
    with TestClient(app) as api:
        response = api.get(path)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        compact = response.json()['data']
        assert 'meta' not in compact
        expanded = api.get(path + '?meta=true').json()['data']
        metadata = expanded.pop('meta')
        assert expanded == compact
        assert api.get(path + '?meta=false').json()['data'] == compact
        assert api.get(path + '?meta=invalid').status_code == 422
        assert 'unit' not in str(compact) and 'source_url' not in str(compact)
        descriptions = metadata['solar_wind'] if kind == 'summary' else metadata['series']
        assert next(iter(descriptions.values()))['unit'] in ('nT', '')
        if kind == 'solar-wind/latest':
            assert compact['series']['bz']['latest']['value'] is None
            assert compact['series']['bz']['latest']['quality'] == 'flagged'
            assert compact['series']['bz']['status'] == 'stale'
            assert compact['series']['bz']['stale_after_seconds'] == 600
            assert 'provider_quality' not in compact['series']['bz']['latest']
        if kind.endswith('history'):
            assert next(iter(compact['series'].values()))['coverage']['gaps'][0]['from'] == TIME
            assert 'from_' not in compact
        if kind == 'summary':
            assert compact['changes_1h']['bz']['value'] is None
            assert compact['southward_bz']['status'] == 'lower_bound'
        spec = api.get('/openapi.json').json()
        parameter = next(p for p in spec['paths'][path]['get']['parameters'] if p['name'] == 'meta')
        assert parameter['schema']['default'] is False
    assert payload == original  # Projection must not strip the owner's data in place.
    assert all('meta' not in (call.args[1] if len(call.args) > 1 else {}) for call in read.await_args_list)


def test_aggregate_history_keeps_completeness_and_source_changes(monkeypatch):
    payload = observation_payload('solar-wind/history')
    payload.update(resolution_seconds=300, evaluated_from=TIME, evaluated_to=END, aggregation_version=1)
    series = payload['series']['bz']
    series.update(resolution_seconds=300, aggregation='mean_min_max', processing=dict(
        unavailable_buckets=2, recalculation_pending_buckets=1, available_buckets=10, expected_buckets=12))
    series['coverage'] = {**COVERAGE, 'basis': 'available_aggregate_windows'}
    series['points'] = [{**SAMPLE, 'value': -2, 'min': -5, 'max': 1, 'count': 3, 'expected_count': 5,
                         'coverage_percent': 60, 'recalculation_pending': True, 'last_spacecraft': 'ACE',
                         'source_changes': 1, 'negative_count': 2}]
    monkeypatch.setattr(solar_wind, 'read_observations', AsyncMock(return_value=dict(success=True, data=payload)))
    app = FastAPI()
    app.include_router(solar_wind.router)
    with TestClient(app) as api:
        compact = api.get('/public/observations/solar-wind/history?resolution=5m').json()['data']
        expanded = api.get('/public/observations/solar-wind/history?resolution=5m&meta=true').json()['data']
    assert expanded.pop('meta')['aggregation_version'] == 1
    assert expanded == compact
    assert compact['resolution_seconds'] == 300
    point = compact['series']['bz']['points'][0]
    assert point['value'] == -2 and point['min'] == -5 and point['max'] == 1
    assert point['recalculation_pending'] is True and point['source_changes'] == 1
    assert point['coverage_percent'] == 60 and point['last_spacecraft'] == 'ACE'
    assert compact['series']['bz']['processing']['unavailable_buckets'] == 2


@pytest.mark.parametrize('response,status', [
    (JSONResponse(status_code=422, content={'detail': 'Invalid interval'}), 422),
    ({'success': False, 'error': {'code': 'NOT_READY', 'message': 'Not ready'}}, 200),
    ({'success': True, 'data': {'bad': 'payload'}}, 503),
])
def test_observation_projection_preserves_errors(monkeypatch, response, status):
    monkeypatch.setattr(solar_wind, 'read_observations', AsyncMock(return_value=response))
    app = FastAPI()
    app.include_router(solar_wind.router)
    with TestClient(app) as api:
        assert api.get('/public/observations/solar-wind/latest?meta=true').status_code == status


@pytest.mark.parametrize('target,product', forecast_products.PRODUCTS.items())
@pytest.mark.parametrize('suffix', ['', '/metrics'])
def test_forecast_and_metrics_metadata_for_every_product(monkeypatch, target, product, suffix):
    import pandas as pd
    issue = pd.Timestamp(TIME)
    frames = []
    for variable in product.variables:
        row = dict(issue_time=issue, valid_time=pd.Timestamp(END), lead_hours=1)
        row.update({name: .25 if name.startswith('p_') else 400 for name in forecast_products._required_columns(variable)
                    if name not in row})
        frames.append((variable, pd.DataFrame([row])))
    monkeypatch.setattr(forecast_products, '_load_variable_frames', lambda _: frames)
    monkeypatch.setattr(forecast_products, '_continuous_metrics', lambda *_: None)
    monkeypatch.setattr(forecast_products, '_binary_metrics', lambda *_: [
        forecast_products.BinaryMetricsSeries(threshold=5, by_lead_hour=[])])
    app = FastAPI()
    app.include_router(forecasts.router)
    path = f'/{product.visibility}/forecasts/{target}{suffix}'
    with TestClient(app) as api:
        response = api.get(path)
        assert response.status_code == 200
        compact = response.json()['data']
        expanded = api.get(path + '?meta=true').json()['data']
        metadata = expanded.pop('meta')
        assert compact == expanded and 'meta' not in compact
        assert metadata['variables'] == {v.name: {'unit': v.unit} for v in product.variables}
        assert 'unit' not in str(compact)
        assert api.get(path + '?meta=false').json()['data'] == compact
        assert api.get(path + '?meta=invalid').status_code == 422
