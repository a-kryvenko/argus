"""Shared operation runner for manual commands and isolated worker tasks."""
import asyncio


def invoke(name, args):
    if name == 'check-health':
        from clio.commands.check_collector_health import run
        return run(args)
    if name == 'collect-enlil':
        from clio.commands.collect_enlil import run
    elif name == 'collect-events':
        from clio.commands.collect_events import run
    elif name == 'sdo-cleanup':
        from clio.commands.sdo_images import run
    elif name == 'backfill':
        from clio.commands.backfill_observations import run
    elif name == 'normalize':
        from clio.commands.normalize import run
    elif name == 'collect':
        from clio.commands.collect import run
    else:
        raise ValueError(f'Unknown collection operation: {name}')
    from clio.db.session import dispose_engine
    async def execute():
        try:
            return await run(args)
        finally:
            await dispose_engine()
    return asyncio.run(execute())
