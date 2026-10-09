import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from argus_prophet.demo import (Manifest, atomic_publish, forecast_payload, issue_times,
                                observation_grid, validate_evidence, validate_inputs, HOUR)


def manifest():
    return Manifest.model_validate({
        'event': {'name': 'Test only', 'starts_at': '2026-01-19T19:38:00Z',
                  'source_url': 'https://example.com/event', 'description': 'Synthetic test event'},
        'observation_source': 'Test', 'input_source': 'Test', 'products': ['solar-wind-density'],
        'models': {'plasma_density_quantile': {'sha256': 'a' * 64,
                   'training_end': '2025-01-01T00:00:00Z', 'training_evidence': 'Test only'}},
    })


def test_event_clock_keeps_minutes_but_issues_are_closed_hour_boundaries():
    spec = manifest()
    times = issue_times(spec.event)
    assert len(times) == 97
    assert times[0] == datetime(2026, 1, 15, 19, tzinfo=UTC)
    assert times[-1] == datetime(2026, 1, 19, 19, tzinfo=UTC)
    assert spec.event.starts_at.minute == 38
    validate_evidence(spec)
    spec.models['plasma_density_quantile'].training_end = times[0]
    with pytest.raises(ValueError, match='overlaps'):
        validate_evidence(spec)


def test_evidence_must_cover_every_model():
    spec = manifest()
    spec.products.append('solar-wind-speed')
    with pytest.raises(ValueError, match='exactly'):
        validate_evidence(spec)


def test_future_and_unclosed_observations_and_late_files_are_rejected():
    issue = issue_times(manifest().event)[0]
    inputs = SimpleNamespace(as_of=issue, observations=SimpleNamespace(points=[]), speed_observations=[],
                             density_observations=[], measurements=[], files=[], aia_frames=[], gong=None)
    inputs.speed_observations = [SimpleNamespace(issue_time=issue)]
    with pytest.raises(ValueError, match='unclosed'):
        validate_inputs(inputs, issue)
    inputs.speed_observations[0].issue_time -= timedelta(hours=1)
    inputs.files = [SimpleNamespace(observed_at=issue-timedelta(hours=1), available_at=issue+timedelta(seconds=1))]
    with pytest.raises(ValueError, match='not available'):
        validate_inputs(inputs, issue)
    inputs.files = []
    validate_inputs(inputs, issue)


def frame(issue, lead=1, **kwargs):
    return pd.DataFrame([dict(issue_time=issue, valid_time=issue+timedelta(hours=lead), lead_hours=lead,
                              n_q10=2., n_q50=4., n_q90=6., **kwargs)])


def test_forecast_contract_preserves_missing_values_and_checks_times():
    issue = issue_times(manifest().event)[0]
    source = frame(issue)
    source.loc[0, 'n_q10'] = float('nan')
    source = pd.concat([source, frame(issue, 2)])
    payload = forecast_payload('solar-wind-density', {'plasma_density_quantile': source}, issue)
    assert payload['predictions'][0]['variables']['n']['continuous'] is None
    assert payload['predictions'][1]['variables']['n']['continuous']['q50'] == 4
    source.loc[0, 'issue_time'] = issue+timedelta(hours=1)
    with pytest.raises(ValueError, match='disagree'):
        forecast_payload('solar-wind-density', {'plasma_density_quantile': source}, issue)


def test_duplicate_leads_and_crossed_quantiles_fail():
    issue = issue_times(manifest().event)[0]
    source = frame(issue)
    with pytest.raises(ValueError, match='duplicate'):
        forecast_payload('solar-wind-density', {'plasma_density_quantile': pd.concat([source, source])}, issue)
    source.loc[0, 'n_q10'] = 20
    with pytest.raises(ValueError, match='Unordered'):
        forecast_payload('solar-wind-density', {'plasma_density_quantile': source}, issue)


def test_observation_grid_keeps_holes_and_requires_future_coverage():
    spec = manifest()
    start = issue_times(spec.event)[0]-timedelta(hours=24)
    end = issue_times(spec.event)[-1]+timedelta(hours=96)
    releases = {'solar-wind-density': [{'predictions': [{'valid_time': end.isoformat()}]}]}
    rows = [{'time': start.isoformat(), 'values': {'n': 3}}, {'time': end.isoformat(), 'values': {'n': 4}}]
    grid = observation_grid(rows, spec, releases)
    assert len(grid) == 217
    assert grid[1]['values']['n'] is None
    with pytest.raises(ValueError, match='cover'):
        observation_grid(rows[:1], spec, releases)


def test_invalid_publication_keeps_old_version(tmp_path):
    current = tmp_path / 'current.json'
    atomic_publish({'version': 'one', 'value': 1}, current)
    with pytest.raises(ValueError):
        atomic_publish({'version': 'two', 'value': float('nan')}, current)
    assert json.loads(current.read_text())['version'] == 'one'
    atomic_publish({'version': 'two', 'value': 2}, current)
    assert json.loads(current.read_text())['version'] == 'two'
    assert json.loads((tmp_path / 'one.json').read_text())['value'] == 1


def test_demo_rejects_unavailable_proswin_and_wrong_horizon():
    from common.schemas.forecast_inputs import ForecastInputs, ProswinPrediction
    issue = issue_times(manifest().event)[0]
    inputs = ForecastInputs(as_of=issue, read_at=issue, observations={'points': []})
    row = ProswinPrediction(valid_time=issue+24*HOUR, image_slot=issue-72*HOUR,
                            available_at=issue, value=500, model_version='proswin-fold1-science-v1')
    inputs.proswin_predictions = [row]
    validate_inputs(inputs, issue)
    inputs.proswin_predictions = [row.model_copy(update={'available_at': issue+HOUR})]
    with pytest.raises(ValueError, match='PROSWIN'):validate_inputs(inputs, issue)
    inputs.proswin_predictions = [row.model_copy(update={'image_slot': issue-71*HOUR})]
    with pytest.raises(ValueError, match='PROSWIN'):validate_inputs(inputs, issue)
