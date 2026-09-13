"""Owned periodic jobs with PostgreSQL completion markers and session locks."""
import logging
import signal
import threading
from datetime import UTC, datetime
from collections.abc import Callable

from argus_clio.db.session import get_database_url

JOBS = {'refresh': (60, 730200), 'aggregate': (5, 730201)}
logger = logging.getLogger(__name__)


class JobBusy(RuntimeError):
    pass


def slot_for(job: str, now: datetime) -> datetime:
    minutes, _ = JOBS[job]
    now = now.astimezone(UTC)
    return now.replace(minute=now.minute // minutes * minutes, second=0, microsecond=0)


def execute(job: str, run: Callable[[], None], *, scheduled: bool = False, now=None) -> bool:
    import psycopg
    url = get_database_url()
    with psycopg.connect(dbname=url.database, user=url.username, password=url.password,
                         host=url.host, port=url.port, autocommit=True,
                         options='-csearch_path=clio,pg_catalog,pg_temp') as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(%s)', (JOBS[job][1],)).fetchone()[0]:
            raise JobBusy(f'Clio {job} is already running')
        slot = slot_for(job, now or datetime.now(UTC))
        if scheduled:
            previous = conn.execute('SELECT completed_slot FROM clio.scheduled_job WHERE name=%s', (job,)).fetchone()
            if previous and previous[0] >= slot:
                return False
        run()
        if scheduled:
            conn.execute('''INSERT INTO clio.scheduled_job(name, completed_slot, completed_at)
                VALUES (%s, %s, %s) ON CONFLICT (name) DO UPDATE SET
                completed_slot=EXCLUDED.completed_slot, completed_at=EXCLUDED.completed_at''',
                         (job, slot, datetime.now(UTC)))
        return True


def work(job: str, run: Callable[[], None]):
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    while not stopped.is_set():
        try:
            if execute(job, run, scheduled=True):
                logger.info('Clio %s completed', job)
        except JobBusy:
            logger.info('Clio %s already running; checking again in 60 seconds', job)
        except Exception:
            logger.exception('Clio %s failed; retrying in 60 seconds', job)
            import sentry_sdk
            sentry_sdk.capture_exception()
        stopped.wait(60)
