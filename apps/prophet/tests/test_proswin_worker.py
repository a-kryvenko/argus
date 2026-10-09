from types import SimpleNamespace
import hashlib
import json

import httpx
import pandas as pd

from argus_prophet.services import proswin_worker as worker


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
    monkeypatch.setattr(worker, 'cache_root', lambda: tmp_path)
    monkeypatch.setattr(worker, 'physical_features', lambda *_: [400.] * 63)
    monkeypatch.setattr(worker, 'ObservationInputs', SimpleNamespace(model_validate=lambda value: SimpleNamespace(
        as_of=pd.Timestamp(value['as_of']), read_at=now)))
    def get(client, url, **kwargs):
        if url.endswith('forecast-inputs'):
            return SimpleNamespace(json=lambda: {'as_of': kwargs['params']['as_of']})
        if 'SILSO' in url:
            assert 'headers' not in kwargs
            return SimpleNamespace(content=b'2026;1;2026.0;100;0;1;1\n')
        if url.endswith('sdo-images'):
            return SimpleNamespace(json=lambda: {'data': {'items': [{'metadata': {
                'slot_at': slot.isoformat(), 'observed_at': slot.isoformat(),
                'available_at': slot.isoformat(), 'sha256': sha}}]}})
        state['downloads'] += 1
        if state['fail']:
            raise httpx.ConnectError('offline')
        return SimpleNamespace(content=content)
    monkeypatch.setattr(worker, 'get', get)
    def predict(*_):
        state['predictions'] += 1
        return 500.
    runtime = SimpleNamespace(crop=lambda *_: None, predict=predict)
    worker.cycle(runtime, None, 'https://clio.test', 'secret')
    assert not list((tmp_path/'predictions').glob('*.json'))
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
