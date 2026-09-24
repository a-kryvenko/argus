"""Per-container progress files, independent of shared database/source health."""
import json
import math
import os
from pathlib import Path
import tempfile
from time import monotonic

from argus_clio.services.collection.specs import collector_sources, overdue_after, ATTEMPT_TIMEOUT_SECONDS


def heartbeat_path(collector: str) -> Path:
    return Path(os.getenv('ARGUS_COLLECTOR_HEALTH_DIR', '/tmp')) / f'argus-{collector}-heartbeat.json'


class CollectorHeartbeat:
    def __init__(self, collector: str, path: Path | None = None):
        self.active = True
        self.collector = collector
        self.path = path or heartbeat_path(collector)
        self.sources = {source_id: {'started': None, 'finished': None} for source_id in collector_sources(collector)}
        if not self.sources:
            raise ValueError('Unknown collector')
        self.write()

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replacement keeps a simultaneous healthcheck from reading partial JSON.
        with tempfile.NamedTemporaryFile(mode='w', dir=self.path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump({'collector': self.collector, 'active': self.active, 'sources': self.sources}, file)
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def stop(self):
        self.active = False
        self.write()

    def started(self, source_id: str):
        self.sources[source_id]['started'] = monotonic()
        self.write()

    def finished(self, source_id: str):
        self.sources[source_id]['finished'] = monotonic()
        self.write()


def check_heartbeat(collector: str, path: Path | None = None, now: float | None = None) -> dict:
    now = monotonic() if now is None else now
    expected = collector_sources(collector)
    if not expected:
        return {'healthy': False, 'reason': 'unknown_collector'}
    try:
        payload = json.loads((path or heartbeat_path(collector)).read_text())
        if payload['collector'] != collector:
            raise ValueError('Wrong collector')
        if payload.get('active') is not True:
            return {'healthy': False, 'reason': 'collector_stopped'}
        sources = payload['sources']
        result = {}
        for source_id in expected:
            started, finished = sources[source_id]['started'], sources[source_id]['finished']
            if started is None:
                result[source_id] = 'not_started'
                continue
            if not isinstance(started, (int, float)) or not math.isfinite(started) or started > now:
                raise ValueError('Invalid heartbeat timestamp')
            if finished is not None and (not isinstance(finished, (int, float)) or not math.isfinite(finished) or finished > now):
                raise ValueError('Invalid heartbeat timestamp')
            running = finished is None or started > finished
            result[source_id] = ('stalled' if running and now-started > ATTEMPT_TIMEOUT_SECONDS else
                                 'overdue' if now-started > overdue_after(source_id) else 'ok')
        return {'healthy': all(value == 'ok' for value in result.values()), 'sources': result}
    except (OSError, ValueError, TypeError, KeyError):
        return {'healthy': False, 'reason': 'heartbeat_unavailable'}
