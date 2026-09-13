"""Hourly scheduling and exclusive Prophet writers coordinated by PostgreSQL."""
import logging
import os
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from argus_prophet.db.session import connect, open_connection, writer_session

logger = logging.getLogger(__name__)
# Distinct from the publication transaction lock (736218, 1) and Clio job locks.
GENERATION_LOCK = (736218, 2)
COMPLETED = ('succeeded', 'partial', 'imported')


class GenerationBusy(RuntimeError):
    """Another Prophet writer owns the database session lock."""


def recover_interrupted():
    # Owning the global writer lock means no previous DB writer remains active.
    with connect(writing=True) as conn:
        conn.execute("""UPDATE prophet.forecast_run SET status='interrupted',finished_at=%s,
            error='Previous writer ended without recording completion' WHERE status='running'""", (datetime.now(UTC),))
        conn.execute("""UPDATE prophet.forecast_slot SET status='interrupted',finished_at=%s,
            error='Previous writer ended without recording completion' WHERE status='running'""", (datetime.now(UTC),))


@contextmanager
def generation_lock():
    # Closing this unpooled session releases its lock, including on exceptions.
    # It stays in autocommit between short operations, not idle in a long transaction.
    with open_connection(autocommit=True) as conn:
        if not conn.execute('SELECT pg_try_advisory_lock(%s,%s)', GENERATION_LOCK).fetchone()[0]:
            raise GenerationBusy('Another Prophet writer is running')
        with writer_session(conn):
            recover_interrupted()
            yield


def due_slot(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Scheduler time must be timezone-aware')
    return (now.astimezone(UTC) - timedelta(minutes=10)).replace(minute=0, second=0, microsecond=0)


def run_due(generate: Callable[[datetime], None], now: datetime) -> bool:
    with generation_lock():
        slot = due_slot(now)
        with connect(writing=True) as conn:
            previous = conn.execute('SELECT max(slot) FROM prophet.forecast_slot WHERE status=ANY(%s)',
                                    (list(COMPLETED),)).fetchone()[0]
        if previous is not None and previous >= slot:
            return False
        generate(slot)
        # Completion must have committed with the release, not after this callback.
        with connect(writing=True) as conn:
            result = conn.execute('SELECT status FROM prophet.forecast_slot WHERE slot=%s', (slot,)).fetchone()
            if result is None or result[0] not in COMPLETED:
                raise RuntimeError('Forecast generation did not complete its scheduled slot')
        return True


def legacy_marker_path() -> Path:
    from common.config import get_config
    # Compatibility only for the explicit one-time import, never scheduler state.
    directory = Path(os.getenv('PROPHET_STATE_DIR', str(get_config().workdir / 'data/prophet')))
    return directory / 'last-completed-slot'


def import_schedule(path: Path | None = None, *, now: datetime | None = None) -> bool:
    path = path if path is not None else legacy_marker_path()
    try:
        value = path.read_text().strip()
    except FileNotFoundError:
        logger.warning('No legacy schedule marker at %s; the latest due slot may run once', path)
        return False
    slot = datetime.fromisoformat(value)
    if slot.tzinfo is None or slot.utcoffset() is None:
        raise ValueError('Legacy schedule marker must be timezone-aware')
    slot = slot.astimezone(UTC)
    if slot != slot.replace(minute=0, second=0, microsecond=0) or slot > due_slot(now or datetime.now(UTC)):
        raise ValueError('Legacy schedule marker must be an elapsed hourly slot')
    with connect(writing=True) as conn:
        result = conn.execute("""INSERT INTO prophet.forecast_slot(slot,status,attempts,finished_at)
            VALUES (%s,'imported',0,%s) ON CONFLICT(slot) DO NOTHING""", (slot, datetime.now(UTC)))
    return result.rowcount == 1


def list_slots(limit=20):
    from psycopg.rows import dict_row
    if not 1 <= limit <= 100:
        raise ValueError('limit must be between 1 and 100')
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT slot,status,attempts,started_at,finished_at,error FROM prophet.forecast_slot ORDER BY slot DESC LIMIT %s', (limit,))
        return cursor.fetchall()


def work(generate: Callable[[datetime], None], export: Callable[[], None] | None = None) -> None:
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    while not stopped.is_set():
        try:
            if run_due(generate, datetime.now(UTC)):
                logger.info('Forecast generation completed')
        except GenerationBusy:
            logger.info('Another generation is running; checking again in 60 seconds')
        except Exception:
            logger.exception('Forecast generation failed; retrying in 60 seconds')
            import sentry_sdk
            sentry_sdk.capture_exception()
        if export is not None:
            try:
                with generation_lock():
                    export()
            except GenerationBusy:
                logger.info('CSV export deferred while another writer is running')
            except Exception:
                logger.exception('CSV export failed; retrying in 60 seconds without recalculation')
        stopped.wait(60)
