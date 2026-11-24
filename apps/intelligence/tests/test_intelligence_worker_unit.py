from unittest.mock import AsyncMock
import asyncio
import json
import signal

from intelligence import worker


def test_worker_retries_and_stops_gracefully(monkeypatch, capsys, tmp_path):
    from intelligence import monitor
    from intelligence import density_monitor
    monkeypatch.setattr(density_monitor, 'refresh_profiles', AsyncMock(return_value={'profiles': 0, 'failed': 0}))
    monkeypatch.setattr(monitor, 'refresh_profiles', AsyncMock(return_value={'profiles': 0, 'failed': 0}))
    stop = asyncio.Event()
    heartbeat = tmp_path / 'worker-state.json'
    calls = []
    original = signal.getsignal(signal.SIGTERM)
    async def process(product):
        calls.append(product)
        if len(calls) == 1:
            raise ValueError('secret input')
        stop.set()
        return {'status': 'succeeded'}
    monkeypatch.setattr(worker, 'process_once', process)
    worker.worker(interval=.01, stop=stop, heartbeat_file=heartbeat)
    assert calls == ['solar-wind-speed', 'solar-wind-speed']
    assert signal.getsignal(signal.SIGTERM) == original
    assert 'last_checked_at' in json.loads(heartbeat.read_text())
    output = capsys.readouterr().out
    assert 'secret input' not in output
    assert [json.loads(line)['status'] for line in output.splitlines()] == ['error', 'succeeded']
