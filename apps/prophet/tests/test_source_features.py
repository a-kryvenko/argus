from datetime import UTC, datetime, timedelta
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pandas as pd
import pytest
from uuid import uuid4
from common.data import datamanager as object_files
from common.schemas.files import FileReference

from prophet.services.inputs import solar_indices as service
from prophet.services.inputs import cache as source_cache
from common.schemas.observation_files import RawObservationFile
from forecast.inputs import ForecastInputs
from common.schemas.observation import Observation, ObservationPoint

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def reference(content, *, observation='goes', **kwargs):
    kwargs.setdefault('file', FileReference(id=uuid4(), timestamp=NOW, bucket='argus-private',
        key=observation + '/' + hashlib.sha256(content).hexdigest()))
    return RawObservationFile(observation=observation, observed_at=NOW,
        sha256=hashlib.sha256(content).hexdigest(), source_product=observation+'.live', **kwargs)


def inputs(*refs):
    point = ObservationPoint(issue_time=NOW, bx=1, by=2, bz=-3, v=420, n=5,
                             t=100000, kp=2, dst=-10, ap=5, f10_7=120)
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[point]), files=list(refs))


def test_original_reads_s3_bytes_and_rejects_corrupt_content(monkeypatch):
    content = b'raw observation'
    ref = reference(content)
    read = Mock(return_value=content)
    monkeypatch.setattr(object_files, 'read_file', read)
    assert source_cache.original(ref) == content
    read.return_value = b'corrupt'
    with pytest.raises(ValueError, match='checksum'):
        source_cache.original(ref)


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
    result = service.enrich_inputs(ForecastInputs.model_validate(parsed.model_dump()))
    assert 'gong' not in result.model_dump()
    assert result.observations.points[0].s10 is None


def test_retired_aia_inputs_do_not_download_or_load_sunpy(monkeypatch):
    ref = reference(b'aia',observation='aia_171_nrt')
    monkeypatch.setattr(service, 'original', lambda *args: pytest.fail('Retired AIA must not be downloaded'))
    result = service.enrich_inputs(inputs(ref))
    assert 'aia_frames' not in result.model_dump()



@pytest.mark.parametrize('uploaded_at', [NOW-timedelta(days=1), NOW, NOW+timedelta(hours=1)])
def test_retired_gong_inputs_are_never_downloaded(monkeypatch, uploaded_at):
    ref = reference(b'old magnetogram', observation='gong')
    ref.file.timestamp = uploaded_at
    monkeypatch.setattr(service, 'original', lambda *args: pytest.fail('Retired GONG must not be downloaded'))
    result = service.enrich_inputs(inputs(ref))
    assert 'gong' not in result.model_dump()
    assert result.observations.points[0].v == 420


def test_goes_selection_uses_latest_stored_file_without_availability_cutoff():
    first = reference(b'first')
    latest = reference(b'latest', file=FileReference(id=uuid4(), timestamp=NOW+timedelta(hours=1),
        bucket='argus-private', key='goes/latest.json'))
    assert service.select_sources(inputs(first, latest)) == [latest]
    assert service.select_sources(inputs(latest, first)) == [latest]


def test_goes_enrichment_remains_active_without_gong(calibrated_goes, tmp_path, monkeypatch):
    samples, _ = calibrated_goes
    content = samples.to_json(orient='records', date_format='iso').encode()
    ref = reference(content)
    gong = reference(b'old magnetogram', observation='gong')
    calls = []
    def read(bucket, key, **kwargs):
        calls.append(key)
        assert bucket == 'argus-private' and key == ref.file.key
        return content
    monkeypatch.setattr(object_files, 'read_file', read)
    result = service.enrich_inputs(inputs(gong, ref))
    assert len(calls) == 1
    assert {row.metric for row in result.measurements} == {'s10', 'm10', 'y10'}
    assert result.observations.points[0].s10 == 101.


async def test_legacy_snapshot_drops_gong_features_before_storage(monkeypatch):
    from contextlib import nullcontext
    from prophet.services.releases import recording as runs
    payload = inputs().model_dump(mode='json')
    payload['gong'] = {'feature_version': 'gong-bands-v1', 'features': {'b_lat_old': 10.}}
    parsed = ForecastInputs.model_validate_json(json.dumps(payload))
    connection = Mock(execute=AsyncMock(return_value=Mock(rowcount=1)))
    connection.execute.return_value.rowcount = 1
    monkeypatch.setattr(runs, 'transaction', lambda _: nullcontext(connection))
    await runs.RunRecorder('test-run', object()).snapshot(parsed)
    stored = connection.execute.call_args.args[1]
    assert 'gong' not in stored['payload']
    assert 'gong' not in stored['input_diagnostics']
    assert stored['payload']['observations'] == payload['observations']


pytestmark = pytest.mark.anyio


def test_original_reads_object_reference_without_http(tmp_path, monkeypatch):
    content = b'original GOES samples'
    ref = reference(content, file=FileReference(id=uuid4(), timestamp=NOW,
        bucket='argus-private', key='observations/GOES/example.json'))
    read = Mock(return_value=content)
    monkeypatch.setattr(object_files, 'read_file', read)
    assert source_cache.original(ref) == content
    assert read.call_count == 1


def test_missing_reference_never_falls_back_to_file_http():
    with pytest.raises(ValueError):
        reference(b'old original', file=None)
