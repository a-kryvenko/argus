from datetime import UTC, datetime
from unittest.mock import Mock

import httpx
import pytest
import pandas as pd
from forecast.api import ForecastResult
from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.observation import Observation, ObservationPoint
from argus_prophet.services import inputs as observations

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


def test_reads_versioned_owner_contract(monkeypatch):
    def handler(request):
        assert request.url.path == '/internal/v1/observations/forecast-inputs'
        assert request.headers['Authorization'] == 'Bearer test-token'
        assert request.url.params['as_of'] == NOW.isoformat()
        return httpx.Response(200, json=stored_inputs().model_dump(mode='json'))
    configure(monkeypatch, handler)
    assert observations.load_inputs(NOW).observations.points[0].v == 420


@pytest.mark.parametrize('status', [401, 503])
def test_owner_failure_is_propagated_without_fallback(monkeypatch, status):
    configure(monkeypatch, lambda _: httpx.Response(status))
    with pytest.raises(httpx.HTTPStatusError):
        observations.load_inputs(NOW)


def test_empty_inputs_fail_without_ingestion(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['observations']['points'] = []
    configure(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(RuntimeError, match='check ingestion'):
        observations.load_inputs(NOW)


def test_rejects_incompatible_contract(monkeypatch):
    payload = stored_inputs().model_dump(mode='json')
    payload['schema_version'] = 2
    configure(monkeypatch, lambda _: httpx.Response(200, json=payload))
    with pytest.raises(ValueError):
        observations.load_inputs(NOW)


def test_verification_reads_raw_browse_pages_and_excludes_end(monkeypatch):
    from argus_prophet.services.verification import read_targets
    times = pd.date_range('2026-01-01', periods=201, freq='min', tz='UTC')
    rows = [{'observed_at': time.isoformat(), 'metric': 'v', 'value': 400.} for time in times]
    pages = []
    def handler(request):
        assert request.url.path == '/internal/v1/observations/browse'
        assert request.headers['Authorization'] == 'Bearer test-token'
        assert request.url.params['kind'] == 'raw'
        assert request.url.params['order'] == 'asc'
        assert request.url.params['page_size'] == '200'
        assert request.url.params['start'] == times[0].isoformat()
        assert request.url.params['end'] == times[-1].isoformat()
        page = int(request.url.params['page'])
        pages.append(page)
        return httpx.Response(200, json={'success': True, 'data': {
            'items': rows[(page-1)*200:page*200], 'total': len(rows)}})
    configure(monkeypatch, handler)
    result = read_targets(times[0], times[-1])
    assert pages == [1, 2]
    assert sum(row['sample_count'] for row in result['points']) == 200
    assert all(row['value'] == 400. for row in result['points'])


@pytest.mark.parametrize('failure', ['http', 'changed', 'incomplete'])
def test_verification_rejects_failed_or_inconsistent_pages(monkeypatch, failure):
    from argus_prophet.services.verification import read_targets
    def handler(request):
        page = int(request.url.params['page'])
        if failure == 'http':
            return httpx.Response(503)
        return httpx.Response(200, json={'success': True, 'data': {
            'items': [{'observed_at': NOW.isoformat(), 'metric': 'v', 'value': 400}] if page == 1 else [],
            'total': 3 if page == 2 and failure == 'changed' else 2}})
    configure(monkeypatch, handler)
    with pytest.raises((httpx.HTTPStatusError, ValueError)):
        read_targets(NOW - pd.Timedelta(days=1), NOW)


def test_verification_empty_observations(monkeypatch):
    from argus_prophet.services.verification import read_targets
    configure(monkeypatch, lambda _: httpx.Response(200, json={
        'success': True, 'data': {'items': [], 'total': 0}}))
    assert read_targets(NOW - pd.Timedelta(days=1), NOW)['points'] == []


def test_verification_targets_preserve_gaps_and_derive_before_averaging():
    from argus_prophet.services.verification import targets
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
    from argus_prophet.services.verification import targets
    frame = pd.DataFrame({'observed_at': ['2026-01-01T00:00:00Z'] * 2,
                          'metric': ['bx', 'by'], 'value': [3., 4.]})
    assert targets(frame, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)) == []


def mock_models(monkeypatch):
    from types import SimpleNamespace
    load = Mock(side_effect=lambda service, **_: (SimpleNamespace(registry_name=service.registry_name), {}))
    compute = Mock(side_effect=lambda service, *args, **kwargs:
                   ForecastResult(service.registry_name, pd.DataFrame({'value': [1]}), {}))
    from argus_prophet.services.generation import models
    from forecast import api
    monkeypatch.setattr(models, 'load_model', load)
    monkeypatch.setattr(api, 'calculate_snapshot', compute)
    return compute


@pytest.mark.parametrize('selection,artifacts', [
    ('solar-wind-speed', ['plasma_speed_quantile', 'plasma_speed_threshold']),
    ('geomagnetic-activity', ['kp_threshold', 'ap_quantile']),
    ('hmf', ['hmf_total_threshold', 'hmf_southward_threshold']),
    ('dst', ['dst_quantile']),
    ('solar-wind-density', ['plasma_density_quantile']),
])
def test_selected_product_uses_one_snapshot(monkeypatch, selection, artifacts):
    from argus_prophet.services.generation import calculation as generation
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
    from argus_prophet.services.generation.products import PRODUCTS
    from common.schemas.forecast_release import PRODUCT_ARTIFACTS
    assert set(PRODUCTS) == set(PRODUCT_ARTIFACTS) - {'solar-radiation'}
    for name, product in PRODUCTS.items():
        assert product.artifacts == PRODUCT_ARTIFACTS[name]



def test_status_selections_include_canonical_names_and_historical_aliases():
    from argus_prophet.services.generation.products import attempt_selections
    assert attempt_selections('solar-wind-speed') == ['all', 'solar-wind-speed', 'wind']
    assert attempt_selections('dst') == ['all', 'dst']
    assert attempt_selections('solar-radiation') == []
