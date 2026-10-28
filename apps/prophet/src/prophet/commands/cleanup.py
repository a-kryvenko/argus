import json


async def run(args):
    from common.config import get_config
    from prophet.services.releases.retention import cleanup
    get_config()
    if args.apply:
        from prophet.db.locks import retention_lock
        async with retention_lock() as writer:
            result = await cleanup(apply=True, writer=writer)
    else:
        result = await cleanup()
    print(json.dumps(result, indent=2))
