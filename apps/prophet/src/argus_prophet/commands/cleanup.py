import json


def run(args):
    from common.config import get_config
    from argus_prophet.services.retention import cleanup
    get_config()
    if args.apply:
        from argus_prophet.scheduling.jobs import retention_lock
        with retention_lock():
            result = cleanup(days=args.days, apply=True)
    else:
        result = cleanup(days=args.days)
    print(json.dumps(result, indent=2))
