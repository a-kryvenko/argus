import pytest
import asyncio
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from clio.routers import observation_files as routes

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def test_archive_override_does_not_load_project_environment(tmp_path, monkeypatch):
    from common import config
    from clio.observations.files import archive_root
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    configure = Mock(side_effect=AssertionError('Archive override must not load project environment'))
    monkeypatch.setattr(config, 'get_config', configure)
    assert archive_root('gong') == tmp_path
    configure.assert_not_called()


def test_catalog_reads_causal_metadata_without_features_or_blobs(monkeypatch):
    monkeypatch.setattr(routes, "snapshot_receipt", lambda row: vars(row))
    monkeypatch.setattr(routes, 'aia_files', lambda *args, **kwargs: [])
    session = AsyncMock()
    row = SimpleNamespace(slot_at=NOW, observed_at=NOW, available_at=NOW, sha256='a'*64, source_product='gong.live')
    session.scalars.side_effect = [Mock(all=lambda: [row]), Mock(all=lambda: [])]
    files = asyncio.run(routes.load_observation_files(session, NOW))
    assert len(files) == 1 and files[0].kind == 'gong'
    assert 'features' not in files[0].model_dump()
    for call in session.scalars.call_args_list:
        query = call.args[0].compile(dialect=postgresql.dialect())
        assert query.params['received_at_1'] == NOW
        assert 'fits_gzip' not in str(query) and 'file_observation.features' not in str(query)


def client(monkeypatch, row):
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'secret')
    session = AsyncMock()
    session.scalars.return_value = Mock(first=lambda: row)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_request_db_session] = lambda: session
    return TestClient(app), session


def test_file_read_is_authenticated_and_checksum_verified(tmp_path, monkeypatch):
    content = b'original FITS'
    digest = hashlib.sha256(content).hexdigest()
    (tmp_path / 'source.fits.gz').write_bytes(gzip.compress(content))
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    http, session = client(monkeypatch, SimpleNamespace(raw_path='source.fits.gz'))
    url = '/internal/v1/observations/files/gong/' + digest
    assert http.get(url).status_code == 401
    session.scalars.assert_not_called()
    result = http.get(url, headers={'Authorization': 'Bearer secret'})
    assert result.status_code == 200 and result.content == content
    (tmp_path / 'source.fits.gz').write_bytes(gzip.compress(b'corrupted'))
    assert http.get(url, headers={'Authorization': 'Bearer secret'}).status_code == 503


def test_missing_archived_original_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    http, session = client(monkeypatch, SimpleNamespace(raw_path='missing.gz'))
    response = http.get('/internal/v1/observations/files/gong/' + 'a'*64,
                        headers={'Authorization': 'Bearer secret'})
    assert response.status_code == 503
    session.scalar.assert_not_awaited()


def test_file_read_rejects_paths_outside_archive(tmp_path, monkeypatch):
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    http, _ = client(monkeypatch, SimpleNamespace(raw_path='../outside.fits'))
    assert http.get('/internal/v1/observations/files/gong/' + 'a'*64,
                    headers={'Authorization': 'Bearer secret'}).status_code == 503


@pytest.mark.parametrize('age_days', [0, 39])
def test_aia_uses_shared_sdo_original_and_receipt_without_database(tmp_path, monkeypatch, age_days):
    import numpy as np
    from common.sdo_images import save_image, save_original
    now = datetime.now(UTC)
    slot = now.replace(minute=0, second=0, microsecond=0) - timedelta(days=age_days)
    content = b'native AIA FITS'
    digest = hashlib.sha256(content).hexdigest()
    metadata = dict(observed_at=slot.isoformat(), available_at=now.isoformat(), source='nrt',
                    preprocessing='sdo-area-mean-unregistered-v1', units='DN', sha256=digest)
    monkeypatch.setenv('ARGUS_SDO_ARCHIVE', str(tmp_path))
    save_image(tmp_path, slot, 'aia193', np.ones((512, 512), np.float32), metadata, now=now)
    assert routes.aia_files(now) == []  # A reduced image cannot stand in for native FITS.
    save_original(tmp_path, slot, 'aia193', content, now=now)
    refs = routes.aia_files(now)
    assert len(refs) == 1 and refs[0].sha256 == digest and refs[0].available_at == now
    assert routes.aia_files(slot) == []
    http, session = client(monkeypatch, None)
    response = http.get('/internal/v1/observations/files/aia/' + digest,
                        headers={'Authorization': 'Bearer secret'})
    assert response.status_code == 200 and response.content == content
    direct = http.get('/internal/v1/observations/files/aia/' + digest,
                      params={'slot_at': slot.isoformat()}, headers={'Authorization': 'Bearer secret'})
    assert direct.status_code == 200 and direct.content == content
    missing = http.get('/internal/v1/observations/files/aia/' + digest,
                       params={'slot_at': (slot-timedelta(hours=1)).isoformat()},
                       headers={'Authorization': 'Bearer secret'})
    assert missing.status_code == 404
    session.scalars.assert_not_awaited()


@pytest.mark.parametrize('channel', ['aia171', 'aia211'])
def test_proswin_channel_originals(tmp_path, monkeypatch, channel):
    import numpy as np
    from common.sdo_images import save_image, save_original
    now = datetime.now(UTC)
    slot = now.replace(minute=0, second=0, microsecond=0)
    content = channel.encode()
    digest = hashlib.sha256(content).hexdigest()
    metadata = dict(observed_at=slot.isoformat(), available_at=now.isoformat(), source='nrt',
                    preprocessing='sdo-area-mean-unregistered-v1', units='DN', sha256=digest)
    monkeypatch.setenv('ARGUS_SDO_ARCHIVE', str(tmp_path))
    save_image(tmp_path, slot, channel, np.ones((512, 512), np.float32), metadata, now=now)
    save_original(tmp_path, slot, channel, content, now=now)
    http, _ = client(monkeypatch, None)
    url = '/internal/v1/observations/files/aia/' + digest
    headers = {'Authorization': 'Bearer secret'}
    catalog = http.get('/internal/v1/observations/files', params={
        'kind': 'aia', 'channel': channel, 'start': slot.isoformat(),
        'end': (slot+timedelta(hours=1)).isoformat(), 'as_of': now.isoformat()}, headers=headers)
    assert catalog.status_code == 200
    assert catalog.json()['data'][0]['sha256'] == digest
    assert catalog.json()['data'][0]['source_product'] == 'aia.nrt_' + channel[3:]
    response = http.get(url, params={'slot_at': slot.isoformat(), 'channel': channel}, headers=headers)
    assert response.status_code == 200 and response.content == content
    assert http.get(url, params={'slot_at': slot.isoformat()}, headers=headers).status_code == 404
    assert http.get(url, params={'channel': 'arbitrary'}, headers=headers).status_code == 422


def test_catalog_http_uses_requested_kind_interval_and_cutoff(monkeypatch):
    http, session = client(monkeypatch, None)
    session.scalars.return_value = Mock(all=lambda: [])
    start = NOW-timedelta(days=7)
    result = http.get('/internal/v1/observations/files', params={
        'kind': 'goes', 'start': start.isoformat(), 'end': NOW.isoformat(), 'as_of': NOW.isoformat()},
        headers={'Authorization': 'Bearer secret'})
    assert result.status_code == 200 and result.json()['data'] == []
    session.scalars.assert_awaited_once()
    query = session.scalars.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'clio.file_observation' in str(query)
    assert query.params['kind_1'] == 'goes'
    assert query.params['source_key_1'] == start.replace(hour=0).isoformat()
    assert query.params['source_key_2'] == (NOW+timedelta(hours=1)).isoformat()
    assert query.params['received_at_1'] == NOW
    assert http.get('/internal/v1/observations/files', params={
        'kind': 'goes', 'start': NOW.isoformat(), 'end': start.isoformat(), 'as_of': NOW.isoformat()},
        headers={'Authorization': 'Bearer secret'}).status_code == 422
