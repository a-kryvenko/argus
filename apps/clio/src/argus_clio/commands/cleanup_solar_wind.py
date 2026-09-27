"""Report or delete verified solar wind raw hours older than 90 days."""
import json
from argus_clio.services.solar_wind.retention import cleanup


async def run(args):
    result = await cleanup(apply=args.apply, retention_days=args.retention_days, limit=args.limit, start=args.start)
    if args.json:
        print(json.dumps(result, default=lambda value: value.isoformat(), indent=2))
    else:
        print(f"Raw history cleanup: {result['mode']} · complete hours before {result['cutoff'].isoformat()}")
        print(f"Examined {result['examined_hours']} source hours; eligible rows: {result['eligible_rows']}; "
              f"deleted rows: {result['deleted_rows']}; skipped hours: {result['skipped_hours']}.")
        for hour in result['hours']:
            print(f"{hour['kind']} {hour['hour'].isoformat()} {hour['status']} "
                  f"{hour.get('reason', '')} rows={hour['rows']}")
        print('Aggregates are retained. Re-run for the next batch; use --from to inspect later hours if earlier ones remain blocked.')
    return int(result['skipped_hours'] > 0)
