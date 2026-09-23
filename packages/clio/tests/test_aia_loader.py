from datetime import UTC,datetime,timedelta
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from clio.dataloaders import aia


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
