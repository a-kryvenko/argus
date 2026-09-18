"""Calculation and byte storage preserve the published artifact representation."""
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from common.adapters import forecast_to_dataframe
from common.exceptions import ConfigurationException
from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.forecast_release import ForecastArtifact
from common.schemas.observation import Observation, ObservationPoint
from forecast.api import calculate_forecast
from forecast_core.api import DstFS
from argus_prophet import generation, ledger
from argus_prophet.models import load_model

ISSUE = datetime(2026, 9, 18, 12, tzinfo=UTC)


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)


@pytest.fixture
def calculation_setup(tmp_path, monkeypatch):
    bundle = {'lead_hours': 2, 'buckets': [(2, 'short')], 'feature_columns': ['v', 'lead_hours'],
              'models': {('short', q): ConstantModel(value)
                         for q, value in zip(('q10', 'q50', 'q90'), (-20., -10., 0.))}}
    path = tmp_path / 'data/models/dst.joblib'
    path.parent.mkdir(parents=True)
    joblib.dump(bundle, path)
    # No forecast output path is required to calculate or store a forecast.
    config = SimpleNamespace(workdir=tmp_path, models_registry={'models': {'dst_quantile': {'model': 'dst'}}})
    monkeypatch.setattr(generation, 'get_config', lambda: config)
    points = [ObservationPoint(issue_time=ISSUE - timedelta(hours=8-i),
                               bx=1, by=2, bz=-3, v=420, n=5, t=100000,
                               kp=2, dst=-10, ap=5, f10_7=120) for i in range(8)]
    inputs = ForecastInputs(as_of=ISSUE, read_at=ISSUE, observations=Observation(points=points))
    return config, path, inputs, bundle


def test_loaded_model_and_pure_calculation_match_previous_csv(calculation_setup, tmp_path):
    config, path, inputs, bundle = calculation_setup
    service, metadata = load_model(DstFS, workdir=config.workdir, registry=config.models_registry['models'])
    assert metadata == {'registry_name': 'dst_quantile', 'model': 'dst',
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = calculate_forecast(service, inputs.observations, issue_time=ISSUE, model_info=metadata)
    expected = forecast_to_dataframe(DstFS(bundle).forecast(inputs.observations, issue_time=ISSUE))
    pd.testing.assert_frame_equal(result.frame, expected)
    assert result.frame.dst_q50.tolist() == [-10., -10.]
    assert result.model_info == metadata and result.name == 'dst_quantile'
    assert not (tmp_path / 'data/forecast').exists()

    # Reference serialization used before this refactor: DataFrame -> file -> bytes.
    reference = tmp_path / 'reference.csv'
    expected.to_csv(reference, index=False)
    recorder = Mock()
    generation.calculate('dst', inputs=inputs, recorder=recorder)
    name, content, info, rows, columns = recorder.store.call_args.args
    assert content == reference.read_bytes()
    assert info == metadata
    artifact = ForecastArtifact(name=name, csv_text=content.decode('utf-8'),
                                sha256=hashlib.sha256(content).hexdigest(),
                                row_count=rows, columns=columns, model_info=info)
    assert artifact.issue_time() == ISSUE
    assert not (tmp_path / 'data/forecast').exists()


def test_storage_failure_stops_run_without_touching_live_exports(calculation_setup, tmp_path):
    _, _, inputs, _ = calculation_setup
    live = tmp_path / 'data/forecast/dst.csv'
    live.parent.mkdir()
    live.write_bytes(b'previous release')
    recorder = Mock()
    recorder.store.side_effect = RuntimeError('database unavailable')
    with pytest.raises(RuntimeError, match='database unavailable'):
        generation.calculate('dst', inputs=inputs, recorder=recorder)
    assert live.read_bytes() == b'previous release'
    assert list(live.parent.iterdir()) == [live]
    recorder.skip.assert_not_called()


@pytest.mark.parametrize('registry', [{}, {'dst_quantile': {'model': 'missing'}}])
def test_model_configuration_errors_are_explicit(tmp_path, registry):
    with pytest.raises(ConfigurationException):
        load_model(DstFS, workdir=tmp_path, registry=registry)


def test_recorder_stores_exact_bytes_without_files(monkeypatch):
    pytest.importorskip('psycopg')
    content = b'issue_time,value\n2026-09-18T12:00:00Z,4\n'
    connection = Mock()
    connect = Mock(return_value=nullcontext(connection))
    monkeypatch.setattr(ledger, 'connect', connect)
    recorder = ledger.RunRecorder('run-id')
    recorder.store('test', content, {'sha256': 'model-hash'}, 1, ['issue_time', 'value'])
    connect.assert_called_once_with(writing=True)
    values = connection.execute.call_args.args[1]
    assert values[:2] == ('run-id', 'test')
    assert gzip.decompress(values[3]) == content
    assert values[4] == hashlib.sha256(content).hexdigest()
    assert values[5] == 1
    assert values[6].obj == ['issue_time', 'value']
    assert values[7].obj == {'sha256': 'model-hash'}


def test_empty_result_is_rejected_before_database_access(monkeypatch):
    connect = Mock()
    monkeypatch.setattr(ledger, 'connect', connect)
    with pytest.raises(ValueError, match='empty forecast'):
        ledger.RunRecorder('run-id').store('test', b'value\n', {}, 0, ['value'])
    connect.assert_not_called()


def test_density_returns_same_grid_and_serialization_without_storage(monkeypatch, tmp_path):
    from common import config
    from argus_prophet.services import density_forecast
    from forecast_core.api import AtmosphericDensityForecastService

    drivers = pd.DataFrame([{
        'valid_time': ISSUE + timedelta(hours=lead), 'observed_at': ISSUE,
        'f10_7': 140., 's10': 115., 'm10': 120., 'y10': 135.,
        'f10_7_81mean': 125., 's10_81mean': 110., 'm10_81mean': 105., 'y10_81mean': 112.,
        'dtc': 80., 'background_method': 'trailing_81_daily_values',
        'background_interpolated_days': 0, 'dtc_method': 'test',
        'history_start': ISSUE - timedelta(days=81), 'dtc_observed_at': ISSUE,
    } for lead in (0, 1)])
    grid = AtmosphericDensityForecastService().forecast_grid(
        drivers, altitudes_km=[400], latitudes_deg=[0], longitudes_deg=[0, 90])
    expected = grid.copy()
    expected.insert(0, 'issue_time', ISSUE)
    expected.insert(2, 'lead_hours', [0, 1])
    monkeypatch.setattr(config, 'get_config', lambda: pytest.fail('Calculation must not load configuration'))
    monkeypatch.setattr(density_forecast, 'load_density_drivers', lambda *args: drivers)
    monkeypatch.setattr(AtmosphericDensityForecastService, 'forecast_grid', lambda *args, **kwargs: grid.copy())
    monkeypatch.chdir(tmp_path)
    inputs = ForecastInputs(as_of=ISSUE, read_at=ISSUE, observations=Observation(points=[]))
    result = density_forecast.calculate_density(inputs=inputs, issue_time=ISSUE)
    pd.testing.assert_frame_equal(result.frame, expected)
    assert list(tmp_path.iterdir()) == []
    assert result.model_info == {'backend': 'forecast_core', 'registry_name': 'atmospheric_density',
                                 'issue_time': ISSUE.isoformat()}
    reference = tmp_path / 'reference.csv'
    expected.to_csv(reference, index=False)
    recorder = Mock()
    generation.store_result(recorder, result)
    name, content, info, rows, columns = recorder.store.call_args.args
    assert content == reference.read_bytes()
    assert rows == 2 and columns == list(expected.columns)
    artifact = ForecastArtifact(name=name, csv_text=content.decode('utf-8'),
                                sha256=hashlib.sha256(content).hexdigest(),
                                row_count=rows, columns=columns, model_info=info)
    assert artifact.issue_time() == ISSUE
