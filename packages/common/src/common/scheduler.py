"""Run scheduled jobs with a heartbeat and drain on shutdown."""
import asyncio
import json
import signal
from datetime import UTC, datetime
from pathlib import Path

from apscheduler.events import EVENT_SCHEDULER_SHUTDOWN
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class WorkerScheduler:
    def __init__(self, *, stop=None):
        self.stop = stop if stop is not None else asyncio.Event()
        self.scheduler = AsyncIOScheduler(
            timezone=UTC,
            job_defaults={'misfire_grace_time': None, 'coalesce': True, 'max_instances': 1},
        )
        self.running = set()

    def add_job(self, function, trigger, *, id, args=(), **options):
        async def run():
            if self.stop.is_set():
                return
            task = asyncio.current_task()
            self.running.add(task)
            try:
                await function(*args)
            finally:
                self.running.discard(task)

        self.scheduler.add_job(run, trigger, id=id, name=id, **options)

    async def serve(self, heartbeat_file=None):
        loop = asyncio.get_running_loop()
        shutdown = asyncio.Event()
        self.scheduler.add_listener(lambda _: shutdown.set(), EVENT_SCHEDULER_SHUTDOWN)
        previous = {}
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous[sig] = signal.getsignal(sig)
                loop.add_signal_handler(sig, self.stop.set)
            self.scheduler.start()
            while not self.stop.is_set():
                if heartbeat_file is None:
                    await self.stop.wait()
                    break
                write_heartbeat(heartbeat_file)
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=10)
                except TimeoutError:
                    continue
        finally:
            self.stop.set()
            try:
                if self.scheduler.running:
                    self.scheduler.pause()
                    await asyncio.gather(*self.running, return_exceptions=True)
                    self.scheduler.shutdown(wait=False)
                    await shutdown.wait()
            finally:
                for sig, handler in previous.items():
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, handler)


def write_heartbeat(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'last_checked_at': datetime.now(UTC).isoformat()}))
    temporary.replace(path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('heartbeat_file', type=Path)
    path = parser.parse_args().heartbeat_file
    try:
        checked = datetime.fromisoformat(json.loads(path.read_text())['last_checked_at'])
        age = (datetime.now(UTC) - checked).total_seconds()
    except (OSError, KeyError, ValueError, TypeError):
        raise SystemExit(1) from None
    raise SystemExit(0 if 0 <= age < 60 else 1)


if __name__ == '__main__':
    main()
