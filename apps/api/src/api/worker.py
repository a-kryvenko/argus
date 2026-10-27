"""Run project monitoring and Nginx traffic aggregation."""
import asyncio
import signal

from common.db.session import dispose_engine
from api.services.monitoring.edge_traffic import run_traffic_loop
from api.services.monitoring.project import run_monitor_loop


def work():
    asyncio.run(run())


async def run():
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, asyncio.current_task().cancel)
    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(run_monitor_loop())
            tasks.create_task(run_traffic_loop())
    except asyncio.CancelledError:
        pass
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        await dispose_engine()
