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


def mock_models(monkeypatch):
    from argus_prophet.services.generation import calculation as generation
    from types import SimpleNamespace
    load = Mock(side_effect=lambda service, **_: (SimpleNamespace(registry_name=service.registry_name), {}))
    compute = Mock(side_effect=lambda service, *args, **kwargs:
                   ForecastResult(service.registry_name, pd.DataFrame({'value': [1]}), {}))
    monkeypatch.setattr(generation, 'load_model', load)
    monkeypatch.setattr(generation, 'calculate_forecast', compute)
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
    recorder = Mock()
    compute = mock_models(monkeypatch)
    generation.calculate(selection, inputs=inputs, recorder=recorder)
    assert [call.args[0].registry_name for call in compute.call_args_list] == artifacts
    for call in compute.call_args_list:
        assert call.args[1] is inputs.observations
        assert call.kwargs['issue_time'] == NOW
    assert [call.args[0] for call in recorder.store.call_args_list] == artifacts


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


def test_dlinear_models_receive_raw_speed_history(monkeypatch):
    from types import SimpleNamespace
    from common.schemas.forecast_inputs import SpeedObservation
    from argus_prophet.services.generation import calculation as generation
    inputs = stored_inputs()
    inputs.speed_observations = [SpeedObservation(issue_time=NOW, v=399.)]
    compute = mock_models(monkeypatch)
    monkeypatch.setattr(generation, 'load_model', lambda service, **kwargs:
        (SimpleNamespace(registry_name=service.registry_name, _dlinear=object()), {}))
    generation.calculate('solar-wind-speed', inputs=inputs, recorder=Mock())
    for call in compute.call_args_list:
        raw = call.kwargs['speed_history']
        assert raw.v.tolist() == [399.]
        assert raw.issue_time.tolist() == [NOW]


def test_aia_models_receive_owner_features_without_filesystem_access(monkeypatch):
    from types import SimpleNamespace
    from common.schemas.forecast_inputs import AIAFeatureFrame
    from argus_prophet.services.generation import calculation as generation
    inputs = stored_inputs()
    inputs.aia_frames = [AIAFeatureFrame(slot_at=NOW,observed_at=NOW,available_at=NOW,sha256='a'*64,features={'aia_area_sector':.2})]
    compute = mock_models(monkeypatch)
    monkeypatch.setattr(generation,'load_model',lambda service,**kwargs:(SimpleNamespace(registry_name=service.registry_name,_dlinear=object(),uses_aia=True),{}))
    generation.calculate('solar-wind-speed',inputs=inputs,recorder=Mock())
    for call in compute.call_args_list:
        frame=call.kwargs['aia_features']
        assert frame.aia_area_sector.tolist()==[.2]
        assert frame.available_at.tolist()==[NOW]


def test_hourly_imf_uses_clio_snapshot_and_read_time(monkeypatch):
    from types import SimpleNamespace
    from argus_prophet.services.generation import calculation as generation
    inputs = stored_inputs()
    inputs.solar_wind_hourly = {'series': {}}
    total = SimpleNamespace(registry_name='hmf_total_threshold', uses_hourly_imf=True,
                            forecast_hourly=Mock(return_value=pd.DataFrame({'lead_hours':[1]})))
    south = SimpleNamespace(registry_name='hmf_southward_threshold')
    monkeypatch.setattr(generation, 'load_model', Mock(side_effect=[(total,{}),(south,{})]))
    compute = Mock(return_value=ForecastResult(south.registry_name,pd.DataFrame({'lead_hours':[1]}),{}))
    monkeypatch.setattr(generation, 'calculate_forecast', compute)
    recorder = Mock()
    generation.calculate('hmf',inputs=inputs,recorder=recorder)
    total.forecast_hourly.assert_called_once_with(inputs.solar_wind_hourly, issue_time=NOW, as_of=inputs.read_at)
    assert compute.call_count == 1
    assert [c.args[0] for c in recorder.store.call_args_list] == ['hmf_total_threshold','hmf_southward_threshold']
