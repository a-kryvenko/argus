"""One scheduler with temporary executors and independent live lanes."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import signal
import threading
from time import monotonic
from types import SimpleNamespace

from clio.config import load_observation_config
from clio.ingestion.products import OBSERVATIONS
from clio.monitoring.heartbeat import CollectorHeartbeat
from clio.monitoring.specs import collector_sources
from clio.scheduling import jobs, observations
from clio.scheduling.execution import invoke_isolated

logger = logging.getLogger(__name__)
STOP_TIMEOUT_SECONDS = 570
BACKGROUND_LIMIT = 2


@dataclass
class Task:
    name: str
    run: Callable[[], bool]
    interval: float = 60
    background: bool = False
    next_run: float = 0


def tasks_for(config, heartbeats, abort):
    def invoke(name, **kwargs):
        return invoke_isolated(name, SimpleNamespace(**kwargs), abort=abort)

    tasks = []
    for mode, kind in (('live', 'numeric'), ('live', 'file'),
                       ('backfill', 'numeric'), ('backfill', 'file')):
        selected = config.model_copy(update={'observations': {
            m: p for m, p in config.observations.items() if OBSERVATIONS[m].kind == kind}})
        if not selected.observations:
            continue
        tasks.append(observation_task(selected, mode, kind, heartbeats, invoke))

    def normalize():
        return jobs.execute('refresh', lambda: invoke('normalize'), scheduled=True)
    tasks.append(Task('normalize', normalize, background=True))
    if config.sdo_images.enabled:
        for mode in ('live', 'warmup', 'cleanup'):
            tasks.append(Task(
                f'sdo-{mode}', lambda mode=mode: (
                    invoke('sdo-cleanup') if mode == 'cleanup' else
                    invoke('collect' if mode == 'live' else 'backfill',
                           metrics=['sdo'], now=datetime.now(UTC), start=None, end=None, scheduled=True)),
                getattr(config.sdo_images, f'{mode}_seconds'), background=mode == 'warmup'))
    return tasks


def observation_task(config, mode, kind, heartbeats, invoke):
    first = mode == 'live'

    def run(metrics, now):
        beats = []
        if mode == 'live' and kind == 'numeric':
            geo = heartbeats.get('geomagnetic')
            wind = heartbeats.get('solar-wind')
            if geo:
                beats.extend((geo, m) for m in metrics if m in geo.sources)
            if wind:
                sources = {product for m in metrics for product in config.observations[m].sources.live}
                beats.extend((wind, f'solar_wind_{kind}') for kind in ('mag', 'plasma')
                             if f'swpc.rtsw_{kind}' in sources)
        for beat, source in beats:
            beat.started(source)
        try:
            return invoke('collect' if mode == 'live' else 'backfill',
                          metrics=metrics, now=now, start=None, end=None, scheduled=True)
        finally:
            for beat, source in beats:
                beat.finished(source)

    def cycle():
        nonlocal first
        result = observations.execute(config, run, mode=mode, force=first)
        first = False
        return result

    interval = min(60, *(getattr(p.schedules, mode).every.total_seconds()
                         for p in config.observations.values()))
    return Task(f'{mode}-{kind}', cycle, interval, mode == 'backfill')


def finish_tasks(running):
    for name, (task, future, started) in list(running.items()):
        if not future.done():
            continue
        try:
            if future.result():
                logger.info('Clio %s completed', name)
        except jobs.JobBusy:
            logger.info('Clio %s already running; retrying later', name)
        except Exception:
            logger.exception('Clio %s failed; retrying later', name)
        task.next_run = monotonic() + task.interval
        del running[name]


def launch_due(tasks, running, pool, stopped=None):
    background = sum(task.background for task, _, _ in running.values())
    now = monotonic()
    # Due background jobs are ordered by waiting time, preventing starvation.
    for task in sorted(tasks, key=lambda t: (t.background, t.next_run)):
        if stopped is not None and stopped.is_set():
            break
        if task.name in running or task.next_run > now:
            continue
        if task.background and background >= BACKGROUND_LIMIT:
            continue
        running[task.name] = (task, pool.submit(task.run), now)
        background += task.background


def work():
    stopped, abort = threading.Event(), threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    heartbeats = {}
    try:
        config = load_observation_config()
        heartbeats['solar-wind'] = CollectorHeartbeat('solar-wind')
        if collector_sources('geomagnetic'):
            heartbeats['geomagnetic'] = CollectorHeartbeat('geomagnetic')
        tasks = tasks_for(config, heartbeats, abort)
        running = {}
        # Reserve capacity for every live lane plus bounded background work. Threads coordinate;
        # provider code and scientific libraries run in spawned processes.
        with ThreadPoolExecutor(max_workers=sum(not task.background for task in tasks) + BACKGROUND_LIMIT,
                                thread_name_prefix='clio-schedule') as pool:
            try:
                while not stopped.is_set():
                    finish_tasks(running)
                    launch_due(tasks, running, pool, stopped)
                    stopped.wait(0.2)
            finally:
                deadline = monotonic() + STOP_TIMEOUT_SECONDS
                while running and monotonic() < deadline:
                    finish_tasks(running)
                    if running:
                        threading.Event().wait(0.2)
                if running:
                    logger.error('Clio shutdown deadline reached; stopping active executors')
                    abort.set()
    finally:
        for beat in heartbeats.values():
            beat.stop()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
