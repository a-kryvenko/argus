"""Backfill configured observations without changing existing records."""
import json

from clio.db.session import get_session_factory
from clio.config import load_observation_config
from clio.ingestion.selection import partition, merge_files, available_files



async def run(args):
    config = load_observation_config()
    numeric, files = partition(config, args.metrics)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.observations.backfill import backfill_selected
        async with get_session_factory()() as session:
            result = await backfill_selected(session, config, numeric, now=args.now, start=args.start, end=args.end,
                                             raise_on_failure=False)
    if files:
        from clio.ingestion.files import collect_file_observations
        result = merge_files(result, await collect_file_observations(
            config, files, mode='backfill', now=args.now, start=args.start, end=args.end))
    print(json.dumps(result, indent=2))
    if (not getattr(args, 'scheduled', False) and result['failed_metrics']
            and not result['downloaded_measurements'] and not available_files(result)):
        raise RuntimeError('No gaps filled; failed observations: ' + ', '.join(result['failed_metrics']))
    return result
