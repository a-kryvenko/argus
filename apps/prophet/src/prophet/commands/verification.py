import json


def run(args):
    from common.config import get_config
    from prophet.services.verification import verify, verification_report
    get_config()
    if args.command == 'verify':
        from prophet.config import load_config
        from prophet.scheduling.execution import supervise, ShutdownRequested
        from prophet.scheduling.jobs import verification_lock
        try:
            with supervise(load_config().shutdown_grace_seconds), verification_lock() as writer:
                result = verify(args.product, writer=writer, days=args.days)
        except ShutdownRequested:
            raise SystemExit(130)
    else:
        result = verification_report(args.product, days=args.days)
    print(json.dumps(result, default=str, indent=2, allow_nan=False))
