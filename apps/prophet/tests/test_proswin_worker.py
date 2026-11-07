from types import SimpleNamespace
import hashlib
import json

import httpx
import pandas as pd

from prophet.services.proswin import worker as worker


def test_worker_retries_transient_network_failure(monkeypatch):
    calls = []
    def request(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise httpx.ConnectError('offline')
        return httpx.Response(200, request=httpx.Request('GET', url))
    monkeypatch.setattr(worker.time, 'sleep', lambda _: None)
    assert worker.get(SimpleNamespace(get=request), 'https://example.test').status_code == 200
    assert len(calls) == 3


def test_failed_slot_resumes_and_completed_slot_is_immutable(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    slot = now.floor('h') - pd.Timedelta(hours=1)
    content = b'fits'
    sha = hashlib.sha256(content).hexdigest()
    state = {'fail': True, 'predictions': 0, 'downloads': 0}
    native = tmp_path / 'native.fits'
    native.write_bytes(content)
    monkeypatch.setattr(worker, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(worker, 'physical_features', lambda *_: [400.] * 63)
    monkeypatch.setattr(worker, 'read_observations', lambda client, url, token, cutoff, **kwargs: SimpleNamespace(as_of=cutoff, read_at=now))
    def get(client, url, **kwargs):
        if 'SILSO' in url:
            assert 'headers' not in kwargs
            return SimpleNamespace(content=b'2026;1;2026.0;100;0;1;1\n')
        if url.endswith('/files'):
            return SimpleNamespace(json=lambda: {'data': [{'file': {'bucket': 'argus-private', 'timestamp': now.isoformat()},
                'observed_at': slot.isoformat(),
                'sha256': sha}]})
        raise AssertionError('Unexpected file HTTP request')
    monkeypatch.setattr(worker, 'get', get)
    def cached(*_):
        state['downloads'] += 1
        if state['fail']:
            raise OSError('offline')
        from contextlib import nullcontext
        return nullcontext(native)
    monkeypatch.setattr(worker, 'temporary_file', cached)
    def predict(*_):
        state['predictions'] += 1
        return 500.
    runtime = SimpleNamespace(crop=lambda *_: None, predict=predict)
    worker.cycle(runtime, None, 'https://clio.test', 'secret')
    assert not list((tmp_path/'predictions').glob('*.json'))
    assert state['downloads'] == 1
    state['fail'] = False
    worker.cycle(runtime, None, 'https://clio.test', 'secret')
    files = list((tmp_path/'predictions').glob('*.json'))
    assert len(files) == 1
    saved = files[0].read_bytes()
    record = json.loads(saved)
    assert pd.Timestamp(record['available_at']) >= now
    assert pd.Timestamp(record['valid_time']) == slot + pd.Timedelta(hours=96)
    def unexpected_model_load():
        raise AssertionError('Cached predictions must not load the neural model')
    worker.cycle(None, None, 'https://clio.test', 'secret', runtime_factory=unexpected_model_load)
    assert files[0].read_bytes() == saved and state['predictions'] == 1
    files[0].write_text('{broken')
    state['loaded'] = False
    def load_runtime():
        state['loaded'] = True
        return runtime
    def physical_after_runtime(*_):
        assert state['loaded'], 'Trusted vendor modules must be registered before physical features'
        return [400.] * 63
    monkeypatch.setattr(worker, 'physical_features', physical_after_runtime)
    worker.cycle(None, None, 'https://clio.test', 'secret', runtime_factory=load_runtime)
    assert json.loads(files[0].read_text())['value'] == 500
    assert state['predictions'] == 2


def test_worker_downloads_fits_from_object_storage_without_clio_file_requests(tmp_path, monkeypatch):
    from uuid import uuid4
    from common.data import datamanager as object_files
    now = pd.Timestamp.now(tz='UTC')
    slot = now.floor('h') - pd.Timedelta(hours=1)
    content = b'fits from S3'
    sha = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(worker, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(worker, 'physical_features', lambda *_: [400.] * 63)
    (tmp_path / 'sunspots.csv').write_bytes(b'2026;1;2026.0;100;0;1;1\n')
    (tmp_path / 'sunspots.json').write_text(json.dumps({'retrieved_at': slot.isoformat(),
        'sha256': hashlib.sha256((tmp_path / 'sunspots.csv').read_bytes()).hexdigest()}))
    reference = dict(id=str(uuid4()), timestamp=now.isoformat(), bucket='argus-private', key='images/SDO/aia171/slot/original.fits')
    def get(client, url, **kwargs):
        assert url.endswith('/files'), 'Clio must only serve a JSON catalog'
        channel = kwargs['params']['observation']
        observed = slot + pd.Timedelta(minutes=-4 if channel == 'aia_171_nrt' else 6)
        return SimpleNamespace(json=lambda: {'data': [{'file': reference,
            'observed_at': observed.isoformat(),
            'sha256': sha}]})
    monkeypatch.setattr(worker, 'get', get)
    monkeypatch.setattr(object_files, 'read_file', lambda bucket, key, **kwargs: content)
    def crop(path, *_):
        assert path.read_bytes() == content
        return None
    runtime = SimpleNamespace(crop=crop, predict=lambda *_: 500.)
    worker.cycle(runtime, None, 'http://clio', 'secret', inputs=SimpleNamespace(as_of=now, read_at=now))
    assert len(list((tmp_path / 'predictions').glob('*.json'))) == 1
    record = json.loads(next((tmp_path / 'predictions').glob('*.json')).read_text())
    assert pd.Timestamp(record['image_slot']) == slot
    assert set(record['image_receipts']) == {'171', '211'}
    assert all('slot_at' not in receipt for receipt in record['image_receipts'].values())
