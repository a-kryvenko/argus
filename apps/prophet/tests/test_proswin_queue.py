from datetime import UTC, datetime, timedelta
import json
import subprocess
import sys
import threading
import time

from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.observation import Observation
from argus_prophet.services import proswin_queue as queue


def test_request_reply_accepts_only_completed_job(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'cache_root', lambda: tmp_path)
    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    def consumer():
        while not list((tmp_path/'jobs').glob('*/request.json')):
            time.sleep(.01)
        job = next((tmp_path/'jobs').glob('*/request.json')).parent
        frozen = json.loads((job/'inputs.json').read_text())
        assert frozen['as_of'] == inputs.model_dump(mode='json')['as_of']
        queue.atomic_write(job/'result.json', json.dumps({'status': 'succeeded',
            'finished_at': datetime.now(UTC).isoformat(), 'predictions': []}).encode())
    thread = threading.Thread(target=consumer)
    thread.start()
    result = queue.prepare(inputs, timeout=2)
    thread.join()
    assert result.proswin_job['status'] == 'succeeded'
    assert result.proswin_ready_at >= now
    assert next((tmp_path/'jobs').glob('*/cancelled')).exists()


def test_unavailable_consumer_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'cache_root', lambda: tmp_path)
    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    result = queue.prepare(inputs, timeout=-5)
    assert result.proswin_job['status'] == 'timeout'
    assert result.proswin_predictions == []


def test_expired_request_does_not_start_model(tmp_path, monkeypatch):
    (tmp_path/'request.json').write_text(json.dumps({'deadline': (datetime.now(UTC)-timedelta(seconds=1)).isoformat()}))
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('started')))
    queue.run_job(tmp_path)


def test_timeout_reaps_child_before_reply(tmp_path, monkeypatch):
    (tmp_path/'request.json').write_text(json.dumps({'deadline': (datetime.now(UTC)+timedelta(seconds=.3)).isoformat()}))
    real = subprocess.Popen
    children = []
    def launch(*args, **kwargs):
        child = real([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    queue.run_job(tmp_path)
    assert children[0].poll() is not None
    assert json.loads((tmp_path/'result.json').read_text())['status'] == 'timeout'


def test_queue_consumer_imports_without_neural_dependencies():
    result = subprocess.run([sys.executable, '-c', '''
import sys
blocked = {'torch', 'torchvision', 'sunpy', 'numpy', 'pandas', 'forecast_core'}
class Guard:
    def find_spec(self, name, *args):
        if name.split('.')[0] in blocked:
            raise AssertionError(name)
sys.meta_path.insert(0, Guard())
from argus_prophet.services.proswin_queue import main
assert callable(main)
assert not blocked.intersection(sys.modules)
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
