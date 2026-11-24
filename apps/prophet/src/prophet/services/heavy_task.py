"""Cross-container exclusion for memory-heavy forecast calculations."""
import asyncio
from contextlib import asynccontextmanager, contextmanager
import fcntl

from prophet.execution import before_dispatch


def lock_path(data_root):
    path = data_root / 'prophet' / 'heavy-task.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def heavy_task(data_root):
    with lock_path(data_root).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@asynccontextmanager
async def wait_for_heavy_task(data_root):
    """Wait for the same process lock without blocking cancellation."""
    with lock_path(data_root).open('a') as stream:
        while True:
            before_dispatch()
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                await asyncio.sleep(0.1)
            else:
                break
        yield
