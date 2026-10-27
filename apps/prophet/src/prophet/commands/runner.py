"""Close shared database resources before the command event loop ends."""
import asyncio

from common.db.session import dispose_engine


def run_async(operation):
    async def execute():
        try:
            return await operation
        finally:
            await dispose_engine()
    return asyncio.run(execute())
