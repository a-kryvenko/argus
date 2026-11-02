import asyncio
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import subprocess
import sys

import pytest
import pandas as pd
from sqlalchemy.dialects import postgresql

from clio.ingestion import gong
from clio.providers import gong as provider
from contextlib import asynccontextmanager

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def test_provider_candidates_use_observed_time_and_bound_range(monkeypatch):
    response = Mock(status_code=200, text='''
        <a href="mrzqs260929t1154c.fits.gz">latest</a>
        <a href="mrzqs260929t1254c.fits.gz">future</a>
        <a href="https://other.example/mrzqs260929t1054c.fits.gz">external</a>
        <a href="mrzqs260928t1154c.fits.gz">old</a>''')
    monkeypatch.setattr('requests.get', Mock(return_value=response))
    result = provider.candidates(NOW-timedelta(hours=3), NOW)
    assert result.to_dict('records') == [dict(timestamp=NOW-timedelta(minutes=6), gong_url=provider.LIVE_URL+'mrzqs260929t1154c.fits.gz')]


def test_archive_404_is_an_unpublished_gap(monkeypatch):
    response = Mock(status_code=404)
    monkeypatch.setattr('requests.get', Mock(return_value=response))
    assert provider.candidates(NOW-timedelta(days=1), NOW, historical=True).empty
    response.raise_for_status.assert_not_called()


def test_compressed_original_has_bounded_expansion(monkeypatch):
    from unittest.mock import MagicMock
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [gzip.compress(b'a'*1000)]
    monkeypatch.setattr('requests.get', Mock(return_value=response))
    monkeypatch.setattr(provider, 'MAX_FITS_BYTES', 100)
    with pytest.raises(ValueError, match='original exceeds'):
        provider.download('https://provider.example/test.fits.gz')


def test_http_read_module_does_not_import_private_code_or_downloader():
    result = subprocess.run([sys.executable, '-c', '''
import sys
import clio.ingestion.gong
assert not any(name.startswith('forecast_core') for name in sys.modules)
assert 'clio.providers.gong' not in sys.modules
assert 'astropy' not in sys.modules
'''], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_original_is_archived_without_features(monkeypatch, tmp_path):
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    original = b'original FITS bytes'
    download = Mock(return_value=original)
    monkeypatch.setattr(provider, 'download', download)
    row = gong.snapshot_values(NOW-timedelta(minutes=6), 'https://provider/file.fits.gz', 'gong.live')
    assert gzip.decompress((tmp_path / row['raw_path']).read_bytes()) == original
    assert row['sha256'] == hashlib.sha256(original).hexdigest()
    assert row['slot_at'] == NOW-timedelta(hours=1)
    assert 'features' not in row and 'fits_gzip' not in row
    assert gong.snapshot_values(NOW-timedelta(minutes=6), 'https://provider/file.fits.gz', 'gong.live') == row
    download.assert_called_once()


def archive_rows(monkeypatch, tmp_path, hours):
    monkeypatch.setenv('ARGUS_GONG_ARCHIVE', str(tmp_path))
    monkeypatch.setattr(provider, 'download', Mock(return_value=b'original FITS'))
    rows = [SimpleNamespace(**gong.snapshot_values(hour+timedelta(minutes=54),
             f'https://provider/{hour:%Y%m%d%H}.fits.gz', 'gong.archive')) for hour in hours]
    session = AsyncMock()
    session.execute.return_value = Mock(scalar_one_or_none=lambda: 1)
    @asynccontextmanager
    async def connection():
        yield session
    monkeypatch.setattr(gong, 'get_db_session', connection)
    monkeypatch.setattr(gong, 'snapshots', AsyncMock(return_value=rows))
    return rows


def test_complete_gong_backfill_does_not_list_or_download_from_provider(monkeypatch, tmp_path):
    from clio.config import load_observation_config
    start = NOW-timedelta(hours=3)
    archive_rows(monkeypatch, tmp_path, [start+timedelta(hours=i) for i in range(3)])
    candidates = Mock(side_effect=AssertionError('Complete archive must not be listed again'))
    download = Mock(side_effect=AssertionError('Complete original must not be downloaded again'))
    monkeypatch.setattr(provider, 'candidates', candidates)
    monkeypatch.setattr(provider, 'download', download)
    result = asyncio.run(gong.collect(load_observation_config(), mode='backfill', now=NOW, start=start, end=NOW))
    assert result['status'] == 'complete'
    assert result['files']['gong']['retained'] == 3
    candidates.assert_not_called()
    download.assert_not_called()


def test_gong_lists_only_days_with_missing_hours_and_downloads_only_gap(monkeypatch, tmp_path):
    from clio.config import load_observation_config
    end = NOW.replace(hour=2)
    start = end-timedelta(hours=6)
    archive_rows(monkeypatch, tmp_path, [start+timedelta(hours=i) for i in range(5)])
    observed = end-timedelta(minutes=6)
    candidates = Mock(return_value=pd.DataFrame({'timestamp': [observed], 'gong_url': ['https://provider/missing.fits.gz']}))
    download = Mock(return_value=b'missing original FITS')
    monkeypatch.setattr(provider, 'candidates', candidates)
    monkeypatch.setattr(provider, 'download', download)
    result = asyncio.run(gong.collect(load_observation_config(), mode='backfill', now=NOW, start=start, end=end))
    assert result['status'] == 'complete'
    candidates.assert_called_once_with(end.replace(hour=0), end, historical=True)
    download.assert_called_once_with('https://provider/missing.fits.gz')


def test_gong_restores_missing_original_without_listing_archive(monkeypatch, tmp_path):
    from clio.config import load_observation_config
    start = NOW-timedelta(hours=1)
    row, = archive_rows(monkeypatch, tmp_path, [start])
    (tmp_path / row.raw_path).unlink()
    candidates = Mock(side_effect=AssertionError('Original URL is already recorded'))
    download = Mock(return_value=b'original FITS')
    monkeypatch.setattr(provider, 'candidates', candidates)
    monkeypatch.setattr(provider, 'download', download)
    result = asyncio.run(gong.collect(load_observation_config(), mode='backfill', now=NOW, start=start, end=NOW))
    assert result['status'] == 'complete'
    assert result['files']['gong']['restored'] == 1
    candidates.assert_not_called()
    download.assert_called_once_with(row.source_url)
