"""Cross-container exclusion for memory-heavy forecast calculations."""
from contextlib import contextmanager
import fcntl


@contextmanager
def heavy_task(data_root):
    path = data_root / 'prophet' / 'heavy-task.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield
