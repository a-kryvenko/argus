"""Single-host scheduling until Prophet gains its database-backed run ledger.

Shared-volume flock serializes all worker and CLI writes to the live CSV files.
A completion marker avoids repeating a successful hourly slot after restart.
This is not a durable forecast publication record and cannot coordinate hosts
without a shared filesystem supporting flock.
"""
import fcntl
import logging
import os
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class GenerationBusy(RuntimeError):
    """Another supported entry point currently owns the output lock."""


def state_directory() -> Path:
    from common.config import get_config
    directory = Path(os.getenv('PROPHET_STATE_DIR', str(get_config().workdir / 'data/prophet')))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@contextmanager
def generation_lock(directory: Path | None = None):
    directory = directory if directory is not None else state_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GenerationBusy('Another Prophet generation is running') from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def due_slot(now: datetime) -> str:
    return (now.astimezone(UTC) - timedelta(minutes=10)).replace(
        minute=0, second=0, microsecond=0,
    ).isoformat()


def run_due(generate: Callable[[], None], directory: Path, now: datetime) -> bool:
    with generation_lock(directory):
        marker = directory / 'last-completed-slot'
        slot = due_slot(now)
        if marker.exists() and marker.read_text().strip() >= slot:
            return False
        generate()
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(slot + '\n')
        temporary.replace(marker)
        return True


def work(generate: Callable[[], None]) -> None:
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    directory = state_directory()
    while not stopped.is_set():
        try:
            if run_due(generate, directory, datetime.now(UTC)):
                logger.info('Forecast generation completed')
        except GenerationBusy:
            logger.info('Another generation is running; checking again in 60 seconds')
        except Exception:
            logger.exception('Forecast generation failed; retrying in 60 seconds')
            import sentry_sdk
            sentry_sdk.capture_exception()
        stopped.wait(60)
