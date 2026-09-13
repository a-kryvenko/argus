"""Report or delete verified solar wind raw hours older than 90 days."""
import argparse
import asyncio
import json
import logging
from argus_clio.commands._runner import run_command
from argus_clio.commands.audit_solar_wind import utc_hour
from argus_clio.db.session import dispose_engine
from argus_clio.services.solar_wind_retention import cleanup


async def run(args):
    try:
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
    finally:
        await dispose_engine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Delete verified raw hours; otherwise report only')
    parser.add_argument('--retention-days', type=int, default=90, help='Minimum 90 days')
    parser.add_argument('--limit', type=int, default=24, help='Source hours to examine, 1–240 (default 24)')
    parser.add_argument('--from', dest='start', type=utc_hour, help='Optional inclusive UTC hour')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.retention_days < 90 or not 1 <= args.limit <= 240:
        parser.error('--retention-days must be at least 90; --limit must be 1–240')
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(asyncio.run(run(args)))


if __name__ == '__main__':
    run_command(main)
