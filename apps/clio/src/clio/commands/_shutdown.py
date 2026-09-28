"""Let collectors finish active requests, then stop without another poll."""
import asyncio
from contextlib import contextmanager
import signal


@contextmanager
def stop_on_signal(enabled):
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = {}
    if enabled:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(
                sig, lambda *_: loop.call_soon_threadsafe(stopped.set))
    try:
        yield stopped
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


async def wait_for_next_poll(stopped, seconds):
    try:
        await asyncio.wait_for(stopped.wait(), timeout=seconds)
    except TimeoutError:
        pass
