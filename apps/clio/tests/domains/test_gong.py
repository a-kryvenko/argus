import asyncio
from datetime import UTC, datetime, timedelta
import gzip
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import subprocess
import sys

import pytest
from sqlalchemy.dialects import postgresql

from clio.domains import gong
from clio.providers import gong as provider

NOW = datetime(2026, 9, 29, 12, tzinfo=UTC)


def test_provider_candidates_use_observed_time_and_bound_range(monkeypatch):
    response = Mock(status_code=200, text='''
        <a href="mrzqs260929t1154c.fits.gz">latest</a>
        <a href="mrzqs260929t1254c.fits.gz">future</a>
        <a href="https://other.example/mrzqs260929t1054c.fits.gz">external</a>
        <a href="mrzqs260928t1154c.fits.gz">old</a>''')
    monkeypatch.setattr('requests.get', Mock(return_value=response))
    result = provider.candidates(NOW-timedelta(hours=3), NOW)
    assert result == [(NOW-timedelta(minutes=6), provider.LIVE_URL+'mrzqs260929t1154c.fits.gz')]


def test_archive_404_is_an_unpublished_gap(monkeypatch):
    response = Mock(status_code=404)
    monkeypatch.setattr('requests.get', Mock(return_value=response))
    assert provider.candidates(NOW-timedelta(days=1), NOW, historical=True) == []
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
import clio.domains.gong
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
    assert (tmp_path / row['raw_path']).read_bytes() == original
    assert row['sha256'] == hashlib.sha256(original).hexdigest()
    assert row['slot_at'] == NOW-timedelta(hours=1)
    assert 'features' not in row and 'fits_gzip' not in row
    assert gong.snapshot_values(NOW-timedelta(minutes=6), 'https://provider/file.fits.gz', 'gong.live') == row
    download.assert_called_once()
