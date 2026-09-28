"""Collect minute solar wind samples once, or continuously with --watch."""
import logging
from time import monotonic

from clio.commands._shutdown import stop_on_signal, wait_for_next_poll
from clio.monitoring.heartbeat import CollectorHeartbeat
from clio.domains.solar_wind.observations import refresh_solar_wind
from clio.monitoring.specs import WIND_POLL_SECONDS

logger = logging.getLogger(__name__)


async def run(args) -> None:
    watch = args.watch
    heartbeat = CollectorHeartbeat('solar-wind') if watch else None
    try:
        with stop_on_signal(watch) as stopped:
            while not stopped.is_set():
                started = monotonic()
                if heartbeat:
                    for source_id in heartbeat.sources:
                        heartbeat.started(source_id)
                try:
                    await refresh_solar_wind()
                except Exception:
                    if not watch:
                        raise
                    logger.exception("Solar wind collection incomplete; retrying next cycle")
                    import sentry_sdk
                    sentry_sdk.capture_exception()
                finally:
                    if heartbeat:
                        for source_id in heartbeat.sources:
                            heartbeat.finished(source_id)
                if not watch:
                    return
                await wait_for_next_poll(stopped, max(1, WIND_POLL_SECONDS - (monotonic() - started)))
    finally:
        if heartbeat:
            heartbeat.stop()
