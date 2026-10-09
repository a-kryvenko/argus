"""Filesystem request/reply queue. The idle consumer imports no model libraries."""
from datetime import UTC, datetime, timedelta
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from argus_prophet.services.proswin_cache import cache_root
from argus_prophet.services.source_cache import atomic_write

logger = logging.getLogger(__name__)
TIMEOUT = 240


def prepare(inputs, timeout=TIMEOUT):
    """Freeze inputs, request inference and accept only this job's reply."""
    job = cache_root() / 'jobs' / uuid.uuid4().hex
    job.mkdir(parents=True)
    deadline = datetime.now(UTC) + timedelta(seconds=timeout)
    source = inputs.model_dump(mode='json', exclude={'aia_frames', 'gong', 'proswin_predictions', 'proswin_ready_at', 'proswin_job'})
    source.update(observations={'points': []}, files=[], measurements=[], solar_wind_hourly=None, density_observations=[])
    atomic_write(job/'inputs.json', json.dumps(source).encode())
    atomic_write(job/'request.json', json.dumps({'deadline': deadline.isoformat()}).encode())
    try:
        while datetime.now(UTC) < deadline + timedelta(seconds=5):
            if (job/'result.json').exists():
                result = json.loads((job/'result.json').read_text())
                inputs.proswin_job = {'id': job.name, **{k:v for k,v in result.items() if k != 'predictions'}}
                if result['status'] == 'succeeded':
                    from common.schemas.forecast_inputs import ProswinPrediction
                    inputs.proswin_predictions = [ProswinPrediction.model_validate(r) for r in result['predictions']]
                    inputs.proswin_ready_at = datetime.fromisoformat(result['finished_at'])
                return inputs
            time.sleep(.25)
        inputs.proswin_job = {'id': job.name, 'status': 'timeout'}
        logger.warning('PROSWIN request timed out; using pre-existing causal cache / DLinear')
        return inputs
    finally:
        atomic_write(job/'cancelled', b'')


def run_job(job):
    request = json.loads((job/'request.json').read_text())
    remaining = (datetime.fromisoformat(request['deadline']) - datetime.now(UTC)).total_seconds()
    if remaining <= 0 or (job/'cancelled').exists():
        atomic_write(job/'result.json', json.dumps({'status': 'expired',
            'finished_at': datetime.now(UTC).isoformat(), 'predictions': []}).encode())
        return
    logger.info('PROSWIN job %s starting', job.name)
    process = subprocess.Popen([sys.executable, '-m', 'argus_prophet.services.proswin_worker',
                                '--job', str(job)], start_new_session=True)
    status = 'failed'
    try:
        while process.poll() is None:
            if (job/'cancelled').exists() or datetime.now(UTC) >= datetime.fromisoformat(request['deadline']):
                status = 'timeout'
                break
            time.sleep(.2)
        else:
            status = 'succeeded' if process.returncode == 0 else 'failed'
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()  # Publish completion only after the neural process is gone.
    result = {'status': status, 'finished_at': datetime.now(UTC).isoformat(), 'predictions': []}
    if status == 'succeeded':
        result['predictions'] = json.loads((job/'predictions.json').read_text())
    atomic_write(job/'result.json', json.dumps(result).encode())
    logger.info('PROSWIN job %s %s; model process exited', job.name, status)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    root = cache_root(); root.mkdir(parents=True, exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            for request in sorted((root/'jobs').glob('*/request.json')):
                job = request.parent
                if (job/'result.json').exists() or (job/'cancelled').exists():
                    continue
                try:
                    run_job(job)
                except Exception:
                    logger.exception('PROSWIN request failed: %s', job.name)
                    atomic_write(job/'result.json', json.dumps({'status': 'failed',
                        'finished_at': datetime.now(UTC).isoformat(), 'predictions': []}).encode())
            cutoff = time.time() - 2*86400
            for request in (root/'jobs').glob('*/request.json'):
                if request.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(request.parent)
            time.sleep(.5)


if __name__ == '__main__':
    main()
