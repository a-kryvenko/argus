"""Report aggregate consistency and retention volume without modifying data."""
import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json

from app.commands._runner import run_command
from app.db.session import dispose_engine, get_session_factory
from app.services.aggregation_audit import audit


def utc_hour(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError('timezone required')
        result = result.astimezone(UTC)
        if result.minute or result.second or result.microsecond:
            raise ValueError('whole UTC hour required')
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


async def check(args, now):
    try:
        async with get_session_factory()() as session:
            result = await audit(session, args.start, args.end, now, args.retention_days, args.detail_limit)
        if args.json:
            print(json.dumps(result, default=lambda value: value.isoformat(), indent=2))
        else:
            counts, storage = result['counts'], result['storage']
            print(f"Aggregate audit: {result['status']} · {args.start.isoformat()} to {args.end.isoformat()}")
            print(f"Checked {counts['checked_buckets']} buckets: {counts['matched_buckets']} match, "
                  f"{counts['missing_buckets']} missing, {counts['mismatched_buckets']} differ; "
                  f"{counts['unverifiable_buckets']} cannot be verified. Pending source hours: {counts['pending_source_hours']}.")
            print(f"Older than {args.retention_days} days: {storage['older_rows']} of {storage['total_rows']} raw rows; "
                  f"{storage['older_payload_bytes']} payload bytes, approximately {storage['estimated_older_allocation_bytes']} allocated bytes.")
            for issue in result['issues']:
                print(f"{issue['kind']} {issue.get('bucket_start', issue['hour']).isoformat()} "
                      f"{issue.get('resolution_seconds', '')} {issue['reason']} "
                      f"{', '.join(issue.get('fields', []))}")
            if result['details_truncated']:
                print(f"Showing {len(result['issues'])} of {result['issue_count']} issues; raise --detail-limit for more.")
            for note in result['notes']:
                print(note)
        return 1 if result['status'] == 'issues_found' else 0
    finally:
        await dispose_engine()


def main():
    now = datetime.now(UTC)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from', dest='start', type=utc_hour, help='Inclusive whole UTC hour, default seven days before to')
    parser.add_argument('--to', dest='end', type=utc_hour, default=now.replace(minute=0, second=0, microsecond=0), help='Exclusive closed UTC hour, default current hour start')
    parser.add_argument('--retention-days', type=int, default=90)
    parser.add_argument('--detail-limit', type=int, default=200)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    args.start = args.start or args.end-timedelta(days=7)
    if not timedelta(0) < args.end-args.start <= timedelta(days=31) or args.end > now:
        parser.error('Choose a positive closed range of at most 31 days')
    if args.retention_days < 1 or args.detail_limit < 1:
        parser.error('--retention-days and --detail-limit must be positive')
    raise SystemExit(asyncio.run(check(args, now)))


if __name__ == '__main__':
    run_command(main)
