import json
import signal
import threading

from intelligence import worker


def test_worker_retries_and_stops_gracefully(monkeypatch, capsys):
    from intelligence import monitor
    from intelligence import density_monitor
    monkeypatch.setattr(density_monitor, 'refresh_profiles', lambda **kwargs: {'profiles': 0, 'failed': 0})
    monkeypatch.setattr(monitor, 'refresh_profiles', lambda **kwargs: {'profiles': 0, 'failed': 0})
    class Stop(threading.Event):
        def wait(self, interval):
            assert interval == 60
            if len(calls) == 2:
                self.set()
    stop = Stop()
    calls = []
    original = signal.getsignal(signal.SIGTERM)
    def process(product):
        calls.append(product)
        if len(calls) == 1:
            raise ValueError('secret input')
        return {'status': 'succeeded'}
    monkeypatch.setattr(worker, 'process_once', process)
    worker.worker(stop=stop)
    assert calls == ['solar-wind-speed', 'solar-wind-speed']
    assert signal.getsignal(signal.SIGTERM) == original
    output = capsys.readouterr().out
    assert 'secret input' not in output
    assert [json.loads(line)['status'] for line in output.splitlines()] == ['error', 'succeeded']
