import json


def run(args):
    from common.config import get_config
    from prophet.services.releases.retention import cleanup
    get_config()
    if args.apply:
        from prophet.scheduling.jobs import retention_lock
        with retention_lock() as writer:
            result = cleanup(days=args.days, apply=True, writer=writer)
    else:
        result = cleanup(days=args.days)
    print(json.dumps(result, indent=2))
