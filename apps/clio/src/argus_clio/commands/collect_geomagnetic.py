"""Collect native Kp/Dst once, or poll independently with --watch."""
import asyncio
import logging
from time import monotonic

from argus_clio.commands._shutdown import stop_on_signal, wait_for_next_poll
from argus_clio.services.collection.heartbeat import CollectorHeartbeat
from argus_clio.services.geomagnetic import ingest_source, refresh_geomagnetic
from clio.dataloaders.geomagnetic_loader import POLL_SECONDS

logger = logging.getLogger(__name__)


async def watch_source(metric: str, heartbeat: CollectorHeartbeat | None = None,
                       stopped: asyncio.Event | None = None) -> None:
    stopped = stopped if stopped is not None else asyncio.Event()
    while not stopped.is_set():
        started = monotonic()
        if heartbeat:
            heartbeat.started(metric)
        try:
            await ingest_source(metric)
        except Exception:
            logger.exception('%s collection failed; retrying next cycle', metric)
            import sentry_sdk
            sentry_sdk.capture_exception()
        finally:
            if heartbeat:
                heartbeat.finished(metric)
        await wait_for_next_poll(stopped, max(1, POLL_SECONDS[metric] - (monotonic() - started)))


async def run(args) -> None:
    watch = args.watch
    heartbeat = None
    try:
        if watch:
            heartbeat = CollectorHeartbeat('geomagnetic')
            with stop_on_signal(True) as stopped:
                await asyncio.gather(*(watch_source(metric, heartbeat, stopped) for metric in POLL_SECONDS))
        else:
            await refresh_geomagnetic()
    finally:
        if heartbeat:
            heartbeat.stop()
