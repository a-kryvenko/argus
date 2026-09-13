from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import importlib

import pandas as pd
import pytest

from argus_prophet import cli
from forecast.ForecastDirector import ForecastDirector


def setup_run(monkeypatch):
    from argus_prophet import ledger, observations
    recorder = Mock(run_id='test-run')
    monkeypatch.setattr(ledger.RunRecorder, 'begin', Mock(return_value=recorder))
    inputs = object()
    monkeypatch.setattr(observations, 'load_inputs', Mock(return_value=inputs))
    command = Mock()
    monkeypatch.setattr(cli.importlib, 'import_module', Mock(return_value=command))
    return recorder, inputs, command


def test_execution_records_snapshot_before_computation(monkeypatch):
    recorder, inputs, command = setup_run(monkeypatch)
    command.main.side_effect = lambda **kwargs: recorder.snapshot.assert_called_once_with(inputs)
    cli.generate('all', trigger='scheduled')
    command.main.assert_called_once_with(inputs=inputs, recorder=recorder)
    recorder.finish.assert_called_once_with()


def test_execution_failure_is_recorded_and_propagated(monkeypatch):
    recorder, _, command = setup_run(monkeypatch)
    failure = ValueError('model failed')
    command.main.side_effect = failure
    with pytest.raises(ValueError, match='model failed'):
        cli.generate('all')
    recorder.finish.assert_called_once_with(error=failure)


def test_snapshot_write_failure_prevents_computation(monkeypatch):
    recorder, _, command = setup_run(monkeypatch)
    recorder.snapshot.side_effect = RuntimeError('database unavailable')
    with pytest.raises(RuntimeError, match='database unavailable'):
        cli.generate('all')
    command.main.assert_not_called()


def frame_writer(monkeypatch, path, stored, written):
    module = importlib.import_module('forecast.ForecastDirector')
    monkeypatch.setattr(module, 'forecast_to_dataframe', lambda _: pd.DataFrame({'issue_time': ['new'], 'value': [2]}))
    service = SimpleNamespace(registry_name='test', forecast=lambda _: None)
    ForecastDirector(stored, written)._build_forecast(path, service, object(), {'sha256': 'model-hash'})


def test_result_is_stored_before_live_csv_replacement(tmp_path, monkeypatch):
    path = tmp_path / 'live.csv'
    path.write_text('issue_time,value\nold,1\n')
    events = []
    def store(name, temporary, metadata, rows, columns):
        assert 'old,1' in path.read_text()
        assert 'new,2' in Path(temporary).read_text()
        assert metadata['sha256'] == 'model-hash' and rows == 1
        events.append('stored')
    def written(name):
        assert 'new,2' in path.read_text()
        events.append('csv')
    frame_writer(monkeypatch, path, store, written)
    assert events == ['stored', 'csv']


def test_database_failure_keeps_previous_live_csv(tmp_path, monkeypatch):
    path = tmp_path / 'live.csv'
    original = 'issue_time,value\nold,1\n'
    path.write_text(original)
    written = Mock()
    with pytest.raises(RuntimeError, match='database unavailable'):
        frame_writer(monkeypatch, path, Mock(side_effect=RuntimeError('database unavailable')), written)
    assert path.read_text() == original
    written.assert_not_called()


def test_recorded_calculation_does_not_publish_live_csv(tmp_path, monkeypatch):
    path = tmp_path / 'live.csv'
    path.write_text('previous release')
    module = importlib.import_module('forecast.ForecastDirector')
    monkeypatch.setattr(module, 'forecast_to_dataframe', lambda _: pd.DataFrame({'value': [2]}))
    stored = Mock()
    service = SimpleNamespace(registry_name='test', forecast=lambda _: None)
    ForecastDirector(on_result=stored, publish_csv=False)._build_forecast(path, service, object())
    stored.assert_called_once()
    assert path.read_text() == 'previous release'
    assert not path.with_name('live.csv.tmp').exists()
