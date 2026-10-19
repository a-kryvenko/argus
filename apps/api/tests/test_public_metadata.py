"""Exercise HTTP serialization: compact and expanded responses share all data."""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.routers import forecasts, observations
from api.services.forecasts import products as forecast_products

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
                    lookback_minutes=75)
    if kind.endswith('latest'):
        return dict(generated_at=TIME, series={'bz': wind} if kind.startswith('solar') else {'kp': index})
    if kind.startswith('geomagnetic'):
        series = {'kp': {**INDEX_META, 'points': [INDEX_SAMPLE], 'coverage': {
            **COVERAGE, 'basis': 'overlapping_intervals', 'resolution_seconds': 10800}}}
    else:
        series = {'bz': {**WIND_META, 'points': [SAMPLE], 'coverage': COVERAGE}}
    return {'from': TIME, 'to': END, 'series': series, 'selection': 'native intervals', 'gap_filling': 'none'}


@pytest.mark.parametrize('kind,selection', [
    ('solar-wind/latest', 'bz'), ('solar-wind/history', 'bz'),
    ('geomagnetic/latest', 'kp'), ('geomagnetic/history', 'kp'),
    ('summary', 'bz,kp'),
])
def test_observation_metadata_is_opt_in_and_data_does_not_change(monkeypatch, kind, selection):
    payload = observation_payload(kind)
    original = deepcopy(payload)
    read = AsyncMock(return_value={'success': True, 'data': payload, 'error': None})
    monkeypatch.setattr(observations, 'read_observations', read)
    app = FastAPI()
    app.include_router(observations.router)
    path = '/observations/history' if kind.endswith('history') else '/observations'
    params = {'observations': selection, 'trends': kind == 'summary'}
    with TestClient(app) as api:
        response = api.get(path, params=params)
        assert response.status_code == 200
        assert response.headers['cache-control'] == 'no-store'
        compact = response.json()['data']
        assert 'unverified' not in str(compact)
        assert 'meta' not in compact
        expanded = api.get(path, params={**params, 'meta': 'true'}).json()['data']
        metadata = expanded.pop('meta')
        assert expanded == compact
        assert api.get(path, params={**params, 'meta': 'false'}).json()['data'] == compact
        assert api.get(path, params={**params, 'meta': 'invalid'}).status_code == 422
        assert 'unit' not in str(compact) and 'source_url' not in str(compact)
        descriptions = metadata['series']
        assert next(iter(descriptions.values()))['unit'] in ('nT', '')
        if kind == 'solar-wind/latest':
            assert compact['bz'] == {'observed_at': TIME, 'value': None, 'quality': 'flagged'}
        if kind.endswith('history'):
            assert next(iter(compact['series'].values()))['coverage']['gaps'][0]['from'] == TIME
            assert 'from_' not in compact
        if kind == 'summary':
            assert compact['changes_1h']['bz'] is None
            assert 'southward_bz' not in compact
        spec = api.get('/openapi.json').json()
        parameter = next(p for p in spec['paths'][path]['get']['parameters'] if p['name'] == 'meta')
        assert parameter['schema']['default'] is False
    assert payload == original
    assert all('meta' not in (call.args[1] if len(call.args) > 1 and call.args[1] else {}) for call in read.await_args_list)


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
    monkeypatch.setattr(observations, 'read_observations', AsyncMock(return_value=dict(success=True, data=payload)))
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as api:
        compact = api.get('/observations/history?observations=bz&resolution=5m').json()['data']
        expanded = api.get('/observations/history?observations=bz&resolution=5m&meta=true').json()['data']
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
    monkeypatch.setattr(observations, 'read_observations', AsyncMock(return_value=response))
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as api:
        assert api.get('/observations?observations=bz&meta=true').status_code == status


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
    path = f'/forecasts/{target}{suffix}'
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


@pytest.mark.parametrize('suffix', ['', '/history'])
def test_unified_observations_select_series_and_metadata(monkeypatch, suffix):
    async def read(resource, params=None):
        if resource == 'history':
            return dict(success=True, data={'points': [{'issue_time': TIME, 'ap': 5, 's10': None}]})
        return dict(success=True, data=observation_payload(resource))

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(observations, 'read_observations', reader)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        selection = 'bz,kp,ap,s10' if suffix else 'bz,kp'
        path = '/observations' + suffix + '?observations=' + selection
        compact = client.get(path)
        assert compact.status_code == 200
        assert compact.headers['cache-control'] == 'no-store'
        compact = compact.json()['data']
        expanded = client.get(path + '&meta=true').json()['data']
        metadata = expanded.pop('meta')
        assert expanded == compact
        if suffix:
            assert set(compact['series']) == {'bz', 'kp'}
            assert set(metadata['series']) == {'bz', 'kp'}
            assert compact['series']['kp']['data_status'] == 'estimated'
            assert compact['points'] == [{'issue_time': TIME, 'ap': 5, 's10': None}]
        else:
            assert {'bz', 'kp'} <= compact.keys()
            assert not {'solar_wind', 'geomagnetic', 'generated_at'} & compact.keys()
            assert not {'issue_time', 'ap', 'f10_7', 's10', 'm10', 'y10'} & compact.keys()
            assert set(compact['bz']) == {'observed_at', 'value', 'quality'}
            assert set(compact['kp']) == {'interval_start', 'interval_end', 'value'}
            assert compact['bz']['value'] is None
            assert compact['bz']['quality'] == 'flagged'
            assert not {'changes_1h', 'southward_bz', 'collection_status', 'bx', 'by', 'v', 'dst'} & compact.keys()
            assert not {'summary', 'status'} & {call.args[0] for call in reader.await_args_list}
            assert set(metadata['series']) == {'bz', 'kp'}
        for selected in ['', 'unknown', 'bz,unknown']:
            assert client.get('/observations' + suffix, params={'observations': selected}).status_code == 422
        for old in ['/public/observations/latest', '/observations/latest', '/observations/summary',
                    '/observations/status', '/observations/solar-wind/latest', '/observations/geomagnetic/history']:
            assert client.get(old).status_code == 404


@pytest.mark.parametrize('meta', [False, True])
def test_current_observations_only_include_trends_when_requested(monkeypatch, meta):
    async def read(resource, params=None):
        assert resource != 'status'
        return dict(success=True, data=observation_payload(resource))

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(observations, 'read_observations', reader)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        params = {'observations': 'bz,kp', 'meta': str(meta).lower()}
        compact = client.get('/observations', params=params)
        assert compact.status_code == 200
        compact = compact.json()['data']
        assert 'changes_1h' not in compact and 'southward_bz' not in compact
        assert 'summary' not in {call.args[0] for call in reader.await_args_list}
        assert client.get('/observations', params={**params, 'trends': 'false'}).json()['data'] == compact
        reader.reset_mock()
        expanded = client.get('/observations', params={**params, 'trends': 'true'})
        assert expanded.status_code == 200
        expanded = expanded.json()['data']
        assert expanded.pop('changes_1h')['bz'] is None
        assert 'southward_bz' not in expanded
        assert expanded == compact
        reader.assert_awaited_once_with('summary')
        assert client.get('/observations?trends=invalid').status_code == 422
        spec = client.get('/openapi.json').json()
        parameter = next(p for p in spec['paths']['/observations']['get']['parameters'] if p['name'] == 'trends')
        assert parameter['schema']['default'] is False


@pytest.mark.parametrize('path', [
    '/observations?observations=bz,kp',
    '/observations?observations=bz,kp&trends=true',
    '/observations/history?observations=bz,kp',
])
def test_normal_observations_omit_quality_without_changing_source_payload(monkeypatch, path):
    originals = {}

    async def read(resource, params=None):
        data = deepcopy(observation_payload(resource))
        groups = [data['solar_wind'], data['geomagnetic']] if resource == 'summary' else [data['series']]
        for group in groups:
            for series in group.values():
                for point in series.get('points', [series.get('latest')]):
                    point['quality'] = 'unverified'
        originals[resource] = (data, deepcopy(data))
        return {'success': True, 'data': data, 'error': None}

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(observations, 'read_observations', reader)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        result = client.get(path)
        assert result.status_code == 200
        data = result.json()['data']
        assert 'quality' not in str(data)
        assert 'unverified' not in str(data)
        assert originals  # Input still uses Clio's internal quality contract.
        for original, snapshot in originals.values():
            assert original == snapshot
            assert 'unverified' in str(original)
        assert 'unverified' not in str(client.get('/openapi.json').json())


@pytest.mark.parametrize('trends', [False, True])
@pytest.mark.parametrize('meta', [False, True])
def test_current_observations_never_load_hourly_snapshot(monkeypatch, trends, meta):
    async def read(resource, params=None):
        assert resource in {'solar-wind/latest', 'geomagnetic/latest', 'summary', 'current-indices'}
        if resource == 'current-indices':
            values = {'ap': {'observed_at': TIME, 'interval_end': END, 'value': 4},
                      'f10_7': {'observed_at': '2026-09-29T20:00:00Z', 'value': 112.5},
                      's10': None, 'm10': None, 'y10': None}
            return dict(success=True, data={'series': {key: values[key] for key in params['metrics'].split(',')}})
        return dict(success=True, data=observation_payload(resource))

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(observations, 'read_observations', reader)
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        result = client.get('/observations', params={'trends': trends, 'meta': meta})
        assert result.status_code == 200
        data = result.json()['data']
        assert data['bz']['observed_at'] == TIME
        assert data['kp']['interval_end'] == END
        assert data['ap'] == {'observed_at': TIME, 'interval_end': END, 'value': 4}
        assert data['f10_7'] == {'observed_at': '2026-09-29T20:00:00Z', 'value': 112.5}
        assert all(data[key] is None for key in ('s10', 'm10', 'y10'))
        assert not {'issue_time', 'generated_at', 'solar_wind', 'geomagnetic', 'points'} & data.keys()
        resources = [call.args[0] for call in reader.await_args_list]
        assert resources == ((['summary'] if trends else ['solar-wind/latest', 'geomagnetic/latest']) + ['current-indices'])
        for metric in ('ap', 'f10_7', 's10', 'm10', 'y10'):
            selected = client.get('/observations', params={'observations': metric})
            assert selected.status_code == 200
            assert selected.json()['data'] == {metric: data[metric]}
        schema = client.get('/openapi.json').json()['components']['schemas']['CurrentObservations']['properties']
        assert not {'issue_time', 'generated_at', 'solar_wind', 'geomagnetic'} & schema.keys()
        assert {'ap', 'f10_7', 's10', 'm10', 'y10'} <= schema.keys()


@pytest.mark.parametrize('value', [60, -2.5, 0, None])
def test_changes_only_return_values_without_metadata(monkeypatch, value):
    data = observation_payload('summary')
    change = {'status': 'available' if value is not None else 'unavailable', 'value': value, 'as_of': TIME,
              'coverage': {'recent': 0.8, 'previous': 0.8}}
    data['changes_1h'] = {'bz': change}
    monkeypatch.setattr(observations, 'read_observations', AsyncMock(return_value=dict(success=True, data=data)))
    app = FastAPI()
    app.include_router(observations.router)
    with TestClient(app) as client:
        result = client.get('/observations?observations=bz&trends=true')
        assert result.status_code == 200
        assert result.json()['data']['changes_1h'] == {'bz': value}


@pytest.mark.parametrize('selection', [None, 'bx,by,t,kp,dst', 'ap,f10_7,s10,m10,y10'])
def test_trends_include_all_requested_variables(monkeypatch, selection):
    wind_and_geomagnetic = (*observations.WIND, *observations.INDICES)
    changes = {key: float(i) for i, key in enumerate(wind_and_geomagnetic)}
    changes.update(ap=2., f10_7=0., s10=None, m10=None, y10=None)

    async def read(resource, params=None):
        if resource == 'current-indices':
            selected = params['metrics'].split(',')
            assert params['trends'] is True
            return dict(success=True, data={
                'series': {key: None for key in selected},
                'changes_1h': {key: changes[key] for key in selected},
            })
        assert resource == 'summary'
        data = observation_payload(resource)
        data['changes_1h'] = {key: {'status': 'available', 'value': changes[key], 'unit': ''}
                              for key in wind_and_geomagnetic}
        return dict(success=True, data=data)

    reader = AsyncMock(side_effect=read)
    monkeypatch.setattr(observations, 'read_observations', reader)
    app = FastAPI()
    app.include_router(observations.router)
    params = {'trends': True}
    if selection is not None:
        params['observations'] = selection
    with TestClient(app) as client:
        response = client.get('/observations', params=params)
        assert response.status_code == 200
        expected = changes if selection is None else {key: changes[key] for key in selection.split(',')}
        assert response.json()['data']['changes_1h'] == expected
