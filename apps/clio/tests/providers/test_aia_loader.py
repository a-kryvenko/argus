from datetime import UTC,datetime,timedelta
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from clio.providers import aia


def test_slots_include_every_hour_not_only_model_cadence():
    now=datetime(2026,9,23,13,27,tzinfo=UTC);slots=aia.hourly_slots(now,1)
    assert len(slots)==25 and slots[-1]==now.replace(minute=0)
    assert {s.hour for s in slots}==set(range(24))
    assert all(b-a==timedelta(hours=1) for a,b in zip(slots,slots[1:]))


def test_missing_snapshot_is_not_filled_and_is_retried(tmp_path,monkeypatch):
    calls=[]
    class Response:
        status_code=404
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def get(url,**kwargs):calls.append(url);return Response()
    monkeypatch.setattr(aia.requests,'get',get);slot=datetime(2025,1,1,1,tzinfo=UTC)
    assert aia.fetch_snapshot(slot,tmp_path) is None
    assert aia.fetch_snapshot(slot,tmp_path) is None
    assert len(calls)==2 and 'H0100/AIA20250101_0100_0193.fits' in calls[0]
    assert not list(tmp_path.rglob('*.fits')) and not list(tmp_path.rglob('*.part'))


def test_original_bytes_and_first_receipt_survive_idempotent_download(tmp_path,monkeypatch):
    content=b'original FITS fixture';slot=datetime(2025,1,1,1,tzinfo=UTC)
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,*args):yield content
    monkeypatch.setattr(aia.requests,'get',lambda *a,**k:Response())
    monkeypatch.setattr(aia,'extract_frame',lambda p:(np.zeros((61,61)),np.ones((61,61)),dict(observed_at=slot.isoformat(),sha256=hashlib.sha256(content).hexdigest(),b0_deg=1.,valid_fraction=1.,carrington_lon=20.)))
    first=aia.fetch_snapshot(slot,tmp_path)
    assert Path(first['raw_path']).read_bytes()==content and datetime.fromisoformat(first['available_at'])>slot
    monkeypatch.setattr(aia.requests,'get',lambda *a,**k:pytest.fail('Downloaded existing snapshot'))
    assert aia.fetch_snapshot(slot,tmp_path)==first


def test_qc_rejected_original_is_retained_without_repeat_download(tmp_path,monkeypatch):
    content=b'QC rejected original';slot=datetime(2025,1,1,1,tzinfo=UTC)
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_content(self,*args):yield content
    monkeypatch.setattr(aia.requests,'get',lambda *a,**k:Response())
    def reject(path):raise ValueError('Nonzero QUALITY')
    monkeypatch.setattr(aia,'extract_frame',reject)
    assert aia.fetch_snapshot(slot,tmp_path) is None
    raw=next(tmp_path.rglob('*.fits'));assert raw.read_bytes()==content
    receipt=json.loads(next(tmp_path.rglob('*.json')).read_text());assert receipt['rejected']
    monkeypatch.setattr(aia.requests,'get',lambda *a,**k:pytest.fail('Repeated QC download'))
    assert aia.fetch_snapshot(slot,tmp_path) is None


@pytest.fixture
def archived(tmp_path, monkeypatch):
    slot = datetime(2025, 1, 1, 1, tzinfo=UTC)
    content = b'original recoverable FITS'
    calls = []

    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, *args): yield content

    def get(*args, **kwargs):
        calls.append(args)
        return Response()

    def extract(path):
        return np.zeros((61, 61)), np.ones((61, 61)), dict(
            observed_at=slot.isoformat(), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            b0_deg=1., valid_fraction=1., carrington_lon=20.)

    monkeypatch.setattr(aia.requests, 'get', get)
    monkeypatch.setattr(aia, 'extract_frame', extract)
    first = aia.fetch_snapshot(slot, tmp_path)
    return slot, content, first, calls


@pytest.mark.parametrize('corrupt', [False, True])
def test_cache_recovery_never_downloads_or_changes_original_receipt(tmp_path, monkeypatch, archived, corrupt):
    slot, content, first, _ = archived
    raw, cache, receipt = aia.snapshot_paths(slot, tmp_path)
    original_receipt = receipt.read_bytes()
    if corrupt:
        cache.write_bytes(b'broken derivative')
    else:
        cache.unlink()
    monkeypatch.setattr(aia.requests, 'get', lambda *a, **k: pytest.fail('Downloaded retained original'))
    assert aia.fetch_snapshot(slot, tmp_path, expected=first) == first
    assert raw.read_bytes() == content
    assert receipt.read_bytes() == original_receipt
    assert aia._valid_cache(cache, first)


@pytest.mark.parametrize('lost_receipt', [False, True])
def test_lost_original_is_restored_with_first_availability(tmp_path, archived, lost_receipt):
    slot, content, first, calls = archived
    raw, cache, receipt = aia.snapshot_paths(slot, tmp_path)
    raw.unlink()
    if lost_receipt:
        receipt.unlink()
    assert aia.fetch_snapshot(slot, tmp_path, expected=first) == first
    assert raw.read_bytes() == content and len(calls) == 2
    assert json.loads(receipt.read_text())['available_at'] == first['available_at']


def test_changed_upstream_cannot_replace_lost_original(tmp_path, monkeypatch, archived):
    slot, _, first, _ = archived
    raw, cache, receipt = aia.snapshot_paths(slot, tmp_path)
    raw.unlink()
    saved_receipt, saved_cache = receipt.read_bytes(), cache.read_bytes()

    def changed(slot, path):
        path.write_bytes(b'revised upstream image')
        return True

    monkeypatch.setattr(aia, '_download', changed)
    with pytest.raises(ValueError, match='SHA256'):
        aia.fetch_snapshot(slot, tmp_path, expected=first)
    assert not raw.exists()
    assert receipt.read_bytes() == saved_receipt and cache.read_bytes() == saved_cache
    assert not list(tmp_path.rglob('*.part'))


def test_corrupted_original_is_reported_without_overwrite(tmp_path, monkeypatch, archived):
    slot, _, first, _ = archived
    raw, _, _ = aia.snapshot_paths(slot, tmp_path)
    raw.write_bytes(b'corrupted original')
    monkeypatch.setattr(aia.requests, 'get', lambda *a, **k: pytest.fail('Overwriting original'))
    with pytest.raises(ValueError, match='SHA256'):
        aia.fetch_snapshot(slot, tmp_path, expected=first)
    assert raw.read_bytes() == b'corrupted original'


def test_database_receipt_conflict_is_not_silently_accepted(tmp_path, archived):
    slot, _, first, _ = archived
    with pytest.raises(ValueError, match='availability/time'):
        aia.fetch_snapshot(slot, tmp_path, expected={**first, 'available_at': slot.isoformat()})


def test_concurrent_requests_share_one_original_and_receipt(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    slot = datetime(2025, 1, 1, 1, tzinfo=UTC)
    calls = []

    def download(slot, path):
        calls.append(slot)
        path.write_bytes(b'one original')
        return True

    monkeypatch.setattr(aia, '_download', download)
    monkeypatch.setattr(aia, 'extract_frame', lambda path: (np.zeros((61, 61)), np.ones((61, 61)), dict(
        observed_at=slot.isoformat(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        b0_deg=1., valid_fraction=1., carrington_lon=20.)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: aia.fetch_snapshot(slot, tmp_path), range(2)))
    assert results[0] == results[1] and calls == [slot]
    assert not list(tmp_path.rglob('*.part'))


def test_nrt_url_and_archive_recovery_preserve_source(tmp_path, monkeypatch):
    slot = datetime(2025, 1, 1, 1, tzinfo=UTC)
    content = b'nrt original'
    calls = []
    class Response:
        status_code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, *args): yield content
    def get(url, **kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(aia.requests, 'get', get)
    monkeypatch.setattr(aia, 'extract_frame', lambda p, **kwargs: (
        np.zeros((61, 61)), np.ones((61, 61)), dict(
            observed_at=slot.isoformat(), sha256=hashlib.sha256(content).hexdigest(),
            b0_deg=1., valid_fraction=1., carrington_lon=20.)))
    first = aia.fetch_snapshot(slot, tmp_path, source_product='aia.nrt_193')
    assert calls == [aia.NRT_URL + '/2025/01/01/H0100/AIA20250101_010000_0193.fits']
    Path(first['raw_path']).unlink()
    restored = aia.fetch_snapshot(slot, tmp_path, expected=first, source_product='aia.synoptic_193')
    assert restored == first
    assert calls[1] == calls[0]


@pytest.mark.parametrize('quality, archive, nrt', [
    (0, True, True), (0x40000000, False, True),
    (-1, False, False), (1, False, False),
    (0x40002000, False, False), (0x80000000, False, False),
])
def test_quality_only_allows_nrt_mode_bit(quality, archive, nrt):
    from clio.domains.aia.extraction import valid_quality
    assert valid_quality(quality) is archive
    assert valid_quality(quality, allow_nrt=True) is nrt


def test_extraction_initializes_runtime_without_writable_home(tmp_path):
    import os
    import subprocess
    import sys
    env = {**os.environ, 'XDG_CONFIG_HOME': str(tmp_path / 'config'),
           'XDG_CACHE_HOME': str(tmp_path / 'cache')}
    result = subprocess.run([sys.executable, '-c', '''
from pathlib import Path
from unittest.mock import patch
with patch.object(Path, 'home', return_value=Path('/')):
    from clio.domains.aia.extraction import extract_frame
    try:
        extract_frame('/nonexistent-aia-runtime-test.fits')
    except ValueError as exc:
        assert 'Did not find any files' in str(exc)
    else:
        raise AssertionError('Expected nonexistent FITS')
    import os
    import sunpy
    manager = Path(sunpy.config.get('downloads', 'remote_data_manager_dir'))
    assert manager.is_relative_to(Path(os.environ['XDG_CACHE_HOME']))
    assert Path(os.environ['XDG_CONFIG_HOME']).is_dir()
'''], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'will be ignored' not in result.stderr
