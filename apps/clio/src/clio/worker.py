"""One scheduler with temporary executors and independent live lanes."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import signal
import threading
from time import monotonic
from types import SimpleNamespace

from clio.config import load_observation_config
from clio.ingestion.products import OBSERVATIONS
from clio.monitoring.heartbeat import CollectorHeartbeat
from clio.monitoring.specs import collector_sources, WIND_POLL_SECONDS
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

    def native():
        beat = heartbeats['solar-wind']
        for source in beat.sources:
            beat.started(source)
        try:
            invoke('native-wind', watch=False)
            return True
        finally:
            for source in beat.sources:
                beat.finished(source)

    tasks = [Task('native-wind', native, WIND_POLL_SECONDS)]
    for mode, kind in (('live', 'numeric'), ('live', 'file'),
                       ('backfill', 'numeric'), ('backfill', 'file')):
        selected = config.model_copy(update={'observations': {
            m: p for m, p in config.observations.items() if OBSERVATIONS[m].kind == kind}})
        if not selected.observations:
            continue
        tasks.append(observation_task(selected, mode, kind, heartbeats, invoke))

    for name, job in (('normalize', 'refresh'), ('aggregate', 'aggregate')):
        def cycle(name=name, job=job):
            return jobs.execute(job, lambda: invoke(name, limit=240), scheduled=True)
        tasks.append(Task(name, cycle, background=True))
    return tasks


def observation_task(config, mode, kind, heartbeats, invoke):
    first = mode == 'live'

    def run(metrics, now):
        beat = heartbeats.get('geomagnetic') if mode == 'live' and kind == 'numeric' else None
        sources = [m for m in metrics if beat and m in beat.sources]
        for source in sources:
            beat.started(source)
        try:
            return invoke('collect' if mode == 'live' else 'backfill',
                          metrics=metrics, now=now, start=None, end=None,
                          scheduled=True, heartbeat=None)
        finally:
            for source in sources:
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
        # Preserve native polling cadence; other schedules poll after completion.
        task.next_run = (max(monotonic(), started + task.interval) if name == 'native-wind'
                         else monotonic() + task.interval)
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
        # Three live lanes plus two background lanes. Threads only coordinate;
        # provider code and scientific libraries run in spawned processes.
        with ThreadPoolExecutor(max_workers=3 + BACKGROUND_LIMIT,
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
