"""Shared operation runner for manual commands and isolated worker tasks."""
import asyncio
import importlib

COMMANDS = {
    'sdo-cleanup': 'sdo_images',
    'backfill': 'backfill_observations', 'normalize': 'normalize',
    'collect': 'collect',
    'check-health': 'check_collector_health',
}


def invoke(name, args):
    module = importlib.import_module(f'clio.commands.{COMMANDS[name]}')
    if name == 'check-health':
        return module.run(args)
    from clio.db.session import dispose_engine
    async def execute():
        try:
            return await module.run(args)
        finally:
            await dispose_engine()
    return asyncio.run(execute())
