from datetime import UTC, datetime, timedelta
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from argus_prophet.services import source_features as service
from common.schemas.forecast_inputs import ForecastInputs, RawObservationFile
from common.schemas.observation import Observation, ObservationPoint

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def reference(content, **kwargs):
    return RawObservationFile(kind='gong', slot_at=NOW, observed_at=NOW,
        available_at=NOW, sha256=hashlib.sha256(content).hexdigest(), source_product='gong.live', **kwargs)


def inputs(*refs):
    point = ObservationPoint(issue_time=NOW, bx=1, by=2, bz=-3, v=420, n=5,
                             t=100000, kp=2, dst=-10, ap=5, f10_7=120)
    return ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[point]), files=list(refs))


def gong_original():
    header = fits.Header({'CRVAL1': 0., 'CRPIX1': 1., 'CDELT1': 1.,
                          'CRVAL2': -.5, 'CRPIX2': 1., 'CDELT2': .02})
    stream = io.BytesIO()
    fits.PrimaryHDU(np.arange(3600, dtype=float).reshape(60, 60), header).writeto(stream)
    return stream.getvalue()


def test_prophet_extracts_real_gong_original_and_preserves_receipt(tmp_path, monkeypatch):
    content = gong_original()
    ref = reference(content)
    calls = []
    monkeypatch.setattr(service, 'cache_root', lambda: tmp_path)
    def handler(request):
        calls.append(request)
        assert request.url.path == '/internal/v1/observations/files/gong/' + ref.sha256
        assert request.headers['Authorization'] == 'Bearer secret'
        return httpx.Response(200, content=content)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = service.enrich_inputs(inputs(ref), client, 'http://clio', 'secret')
        assert result.gong.features['b_lat_0_25_lon_0_25_grad_energy'] >= 0
        assert result.gong.sha256 == ref.sha256
        assert result.gong.available_at == NOW
        assert service.enrich_inputs(inputs(ref), client, 'http://clio', 'secret').gong == result.gong
    assert len(calls) == 1


@pytest.mark.parametrize('problem', ['checksum', 'missing', 'future'])
def test_unusable_original_never_becomes_gong_features(tmp_path, monkeypatch, problem):
    content = gong_original()
    ref = reference(content)
    if problem == 'future':
        ref.available_at = NOW + timedelta(seconds=1)
    monkeypatch.setattr(service, 'cache_root', lambda: tmp_path)
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503) if problem == 'missing' else httpx.Response(200, content=b'changed')
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = service.enrich_inputs(inputs(ref), client, 'http://clio', 'secret')
    assert result.gong is None and result.observations.points[0].v == 420
    if problem == 'future':
        assert not calls


def test_original_cache_detects_corruption_and_refetches(tmp_path, monkeypatch):
    content = gong_original()
    ref = reference(content)
    monkeypatch.setattr(service, 'cache_root', lambda: tmp_path)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))) as client:
        path = service.original(client, 'http://clio', 'secret', ref)
        path.write_bytes(b'broken cache')
        assert service.original(client, 'http://clio', 'secret', ref).read_bytes() == content


@pytest.fixture
def calibrated_goes(tmp_path, monkeypatch):
    from forecast_core.data_pipelines.solar_indices import (
        SolarIndexCalibration, INDEX_FEATURE_COLUMNS, REQUIRED_GOES_COLUMNS, save_solar_index_calibrations)
    artifact = tmp_path / 'calibration.json'
    save_solar_index_calibrations({name: SolarIndexCalibration(
        feature_columns=features, feature_means=(0.,)*len(features), feature_scales=(1.,)*len(features),
        intercept=100., coefficients=(1.,)+(0.,)*(len(features)-1))
        for name, features in INDEX_FEATURE_COLUMNS.items()}, artifact)
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
    from common.schemas.forecast_inputs import ObservationInputs
    source = inputs().model_dump()
    source['gong'] = {'invalid': 'Clio must not supply features'}
    parsed = ObservationInputs.model_validate(source)
    result = service.enrich_inputs(ForecastInputs.model_validate(parsed.model_dump()), None, '', '')
    assert result.gong is None
    assert result.observations.points[0].s10 is None


def test_aia_history_rebuilds_from_clio_after_deleting_cache(tmp_path, monkeypatch):
    import shutil
    from argus_prophet.services.aia import extraction
    monkeypatch.setattr(service, 'cache_root', lambda: tmp_path)
    references, originals, frames = [], {}, {}
    for days, value in [(27, 1.), (0, 2.)]:
        observed = NOW - timedelta(days=days)
        content = f'AIA at {observed}'.encode()
        ref = RawObservationFile(kind='aia', slot_at=observed, observed_at=observed,
            available_at=observed, sha256=hashlib.sha256(content).hexdigest(), source_product='aia.nrt_193')
        originals[ref.sha256] = content
        frames[ref.sha256] = (np.full((61, 61), value), np.ones((61, 61)),
            dict(observed_at=observed.isoformat(), sha256=ref.sha256,
                 b0_deg=0., valid_fraction=1., carrington_lon=0.))
        references.append(ref)
    monkeypatch.setattr(extraction, 'extract_frame', lambda path, **kw: frames[path.stem])
    calls = []
    def handler(request):
        digest = request.url.path.rsplit('/', 1)[1]
        ref = next(r for r in references if r.sha256 == digest)
        assert request.url.params['slot_at'] == ref.slot_at.isoformat()
        calls.append(digest)
        return httpx.Response(200, content=originals[digest])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = service.enrich_inputs(inputs(*references), client, 'http://clio', 'secret')
        assert len(first.aia_frames) == 1
        assert first.aia_frames[0].features['aia_delta_rotation_lat1_lon2'] == pytest.approx(1.)
        shutil.rmtree(tmp_path / 'aia')
        rebuilt = service.enrich_inputs(inputs(*references), client, 'http://clio', 'secret')
        assert rebuilt.aia_frames == first.aia_frames
        assert len(calls) == 4
        # A cache is never an independent source of history.
        assert not service.enrich_inputs(inputs(), client, 'http://clio', 'secret').aia_frames
        future = references[0].model_copy(update={'available_at': NOW + timedelta(hours=1)})
        causal = service.enrich_inputs(inputs(future, references[1]), client, 'http://clio', 'secret')
        assert causal.aia_frames[0].features['aia_delta_rotation_lat1_lon2'] is None


def test_source_cache_prunes_old_files_only(tmp_path, monkeypatch):
    import os
    from datetime import datetime
    monkeypatch.setattr(service, 'cache_root', lambda: tmp_path)
    folder = tmp_path / 'aia'
    folder.mkdir()
    old = folder / 'old.features-v1.npz'
    old.write_bytes(b'old')
    current = folder / 'current.fits'
    current.write_bytes(b'current')
    expired = datetime.now(UTC).timestamp() - timedelta(days=46).total_seconds()
    os.utime(old, (expired, expired))
    service.prune_source_cache()
    assert not old.exists() and current.read_bytes() == b'current'
