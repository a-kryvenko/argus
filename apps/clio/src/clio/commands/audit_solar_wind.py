"""Report aggregate consistency and retention volume without modifying data."""
import json

from clio.db.session import get_session_factory
from clio.domains.solar_wind.audit import audit


async def run(args):
    async with get_session_factory()() as session:
        result = await audit(session, args.start, args.end, args.now, args.retention_days, args.detail_limit)
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
