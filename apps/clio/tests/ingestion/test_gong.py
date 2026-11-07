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

from clio.ingestion.extras import gong
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
import clio.ingestion.extras.gong
assert not any(name.startswith('forecast_core') for name in sys.modules)
assert 'clio.providers.gong' not in sys.modules
assert 'astropy' not in sys.modules
'''], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_original_is_uploaded_and_recorded(storage, monkeypatch):
    from clio.ingestion.utils.file_observations import find
    monkeypatch.setattr(provider, 'download', Mock(return_value=b'original FITS'))
    row = gong.snapshot_values(NOW, 'https://example.test/gong.fits', 'gong.live')
    assert row['sha256'] == hashlib.sha256(b'original FITS').hexdigest()
    assert find('gong')[0]['details']['source_url'] == 'https://example.test/gong.fits'


def test_complete_backfill_uses_db_references_without_source_download(storage, monkeypatch):
    from clio.config import load_observation_config
    config = load_observation_config()
    start = NOW-timedelta(hours=2)
    rows = [dict(observed_at=start+timedelta(hours=i), source_product='gong.archive') for i in range(2)]
    monkeypatch.setattr(gong, 'find', lambda *args, **kwargs: rows)
    monkeypatch.setattr(provider, 'candidates', Mock(side_effect=AssertionError('No provider listing')))
    result = asyncio.run(gong.collect(config, mode='backfill', now=NOW, start=start, end=NOW))
    assert result['status'] == 'complete' and result['files']['gong']['retained'] == 2
