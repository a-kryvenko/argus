from datetime import UTC, datetime, timedelta
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
import pytest

from prophet.services.inputs import solar_indices as service
from prophet.services.inputs import cache as source_cache
from common.schemas.observation_files import RawObservationFile
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation, ObservationPoint

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def reference(content, *, kind='goes', **kwargs):
    return RawObservationFile(kind=kind, slot_at=NOW, observed_at=NOW,
        available_at=NOW, sha256=hashlib.sha256(content).hexdigest(), source_product=kind+'.live', **kwargs)


def inputs(*refs):
    point = ObservationPoint(issue_time=NOW, bx=1, by=2, bz=-3, v=420, n=5,
                             t=100000, kp=2, dst=-10, ap=5, f10_7=120)
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[point]), files=list(refs))


def test_original_cache_detects_corruption_and_refetches(tmp_path, monkeypatch):
    content = b'raw observation'
    ref = reference(content)
    monkeypatch.setattr(source_cache, 'cache_root', lambda: tmp_path)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))) as client:
        path = source_cache.original(client, 'http://clio', 'secret', ref)
        path.write_bytes(b'broken cache')
        assert source_cache.original(client, 'http://clio', 'secret', ref).read_bytes() == content


@pytest.fixture
def calibrated_goes(tmp_path, monkeypatch):
    from forecast_core.data_pipelines.solar_indices import (
        SolarIndexCalibration, INDEX_FEATURE_COLUMNS, REQUIRED_GOES_COLUMNS)
    artifact = tmp_path / 'calibration.json'
    calibrations = {name: SolarIndexCalibration(
        feature_columns=features, feature_means=(0.,)*len(features), feature_scales=(1.,)*len(features),
        intercept=100., coefficients=(1.,)+(0.,)*(len(features)-1))
        for name, features in INDEX_FEATURE_COLUMNS.items()}
    artifact.write_text(json.dumps({
        'version': 1,
        'calibrations': {name: asdict(value) for name, value in calibrations.items()},
    }, allow_nan=False))
    monkeypatch.setattr(service, 'get_config', lambda: SimpleNamespace(workdir=tmp_path,
        models_registry={'models': {'solar_index_calibration': {'calibration_path': artifact.name}}}))
    rows = []
    for timestamp in ('2026-09-28T12:30Z', '2026-09-29T10:30Z', '2026-09-29T11:30Z', '2026-09-29T12:30Z'):
        row = dict.fromkeys(REQUIRED_GOES_COLUMNS, 1.)
        row.update(timestamp=pd.Timestamp(timestamp), goes_euvs_quality_valid=True)
        rows.append(row)
    return pd.DataFrame(rows), artifact


def test_prophet_calibrates_completed_days_and_current_hour(calibrated_goes):
    samples, _ = calibrated_goes
    rows = service.solar_measurements([samples], NOW)
    assert {r.metric for r in rows} == {'s10', 'm10', 'y10'}
    assert {r.observed_at for r in rows} == {NOW, NOW.replace(day=28, hour=12)}
    assert all(r.value == 101 for r in rows)


def test_missing_calibration_keeps_estimates_absent(calibrated_goes):
    samples, artifact = calibrated_goes
    artifact.unlink()
    assert service.solar_measurements([samples], NOW) == []


def test_wire_features_and_legacy_calibrations_are_not_trusted():
    from forecast.observation_inputs import ObservationInputs
    source = inputs().model_dump()
    source['gong'] = {'invalid': 'Clio must not supply features'}
    parsed = ObservationInputs.model_validate(source)
    result = service.enrich_inputs(ForecastInputs.model_validate(parsed.model_dump()), None, '', '')
    assert 'gong' not in result.model_dump()
    assert result.observations.points[0].s10 is None


def test_retired_aia_inputs_do_not_download_or_load_sunpy(monkeypatch):
    ref = RawObservationFile(kind='aia', slot_at=NOW, observed_at=NOW,
        available_at=NOW, sha256='a'*64, source_product='aia.nrt_193')
    monkeypatch.setattr(service, 'original', lambda *args: pytest.fail('Retired AIA must not be downloaded'))
    result = service.enrich_inputs(inputs(ref), None, 'http://clio', 'secret')
    assert 'aia_frames' not in result.model_dump()


def test_source_cache_prunes_old_files_only(tmp_path, monkeypatch):
    import os
    from datetime import datetime
    monkeypatch.setattr(source_cache, 'cache_root', lambda: tmp_path)
    folder = tmp_path / 'aia'
    folder.mkdir()
    old = folder / 'old.features-v1.npz'
    old.write_bytes(b'old')
    current = folder / 'current.fits'
    current.write_bytes(b'current')
    expired = datetime.now(UTC).timestamp() - timedelta(days=46).total_seconds()
    os.utime(old, (expired, expired))
    source_cache.prune_source_cache()
    assert not old.exists() and current.read_bytes() == b'current'


@pytest.mark.parametrize('available_at', [NOW-timedelta(days=1), NOW, NOW+timedelta(hours=1)])
def test_retired_gong_inputs_are_never_downloaded(monkeypatch, available_at):
    ref = reference(b'old magnetogram', kind='gong')
    ref.available_at = available_at
    monkeypatch.setattr(service, 'original', lambda *args: pytest.fail('Retired GONG must not be downloaded'))
    monkeypatch.setattr(service, 'prune_source_cache', lambda: None)
    result = service.enrich_inputs(inputs(ref), None, 'http://clio', 'secret')
    assert 'gong' not in result.model_dump()
    assert result.observations.points[0].v == 420


def test_goes_enrichment_remains_active_without_gong(calibrated_goes, tmp_path, monkeypatch):
    samples, _ = calibrated_goes
    content = samples.to_json(orient='records', date_format='iso').encode()
    ref = reference(content)
    gong = reference(b'old magnetogram', kind='gong')
    calls = []
    monkeypatch.setattr(source_cache, 'cache_root', lambda: tmp_path/'cache')
    def handler(request):
        calls.append(request)
        assert request.url.path == '/internal/v1/observations/files/goes/'+ref.sha256
        assert request.headers['Authorization'] == 'Bearer secret'
        return httpx.Response(200, content=content)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = service.enrich_inputs(inputs(gong, ref), client, 'http://clio', 'secret')
    assert len(calls) == 1
    assert {row.metric for row in result.measurements} == {'s10', 'm10', 'y10'}
    assert result.observations.points[0].s10 == 101.


def test_legacy_snapshot_drops_gong_features_before_storage(monkeypatch):
    from contextlib import nullcontext
    from prophet.services.releases import recording as runs
    payload = inputs().model_dump(mode='json')
    payload['gong'] = {'feature_version': 'gong-bands-v1', 'features': {'b_lat_old': 10.}}
    parsed = ForecastInputs.model_validate_json(json.dumps(payload))
    connection = Mock()
    connection.execute.return_value.rowcount = 1
    monkeypatch.setattr(runs, 'transaction', lambda _: nullcontext(connection))
    runs.RunRecorder('test-run', object()).snapshot(parsed)
    stored = connection.execute.call_args.args[1]
    assert 'gong' not in stored[0].obj
    assert 'gong' not in stored[2].obj
    assert stored[0].obj['observations'] == payload['observations']
