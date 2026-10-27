"""Backfill configured observations without changing existing records."""
import json

from clio.db.session import get_db_session
from clio.config import load_observation_config
from clio.ingestion.selection import partition, merge_files, merge_sdo


async def run(args):
    config = load_observation_config()
    numeric, files = partition(config, args.metrics)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.observations.backfill import backfill_selected
        async with get_db_session() as session:
            result = await backfill_selected(session, config, numeric, now=args.now, start=args.start, end=args.end,
                                             raise_on_failure=False)
    if files:
        from clio.ingestion.files import collect_file_observations
        result = merge_files(result, await collect_file_observations(
            config, files, mode='backfill', now=args.now, start=args.start, end=args.end))
    result = await merge_sdo(result, config, args, mode='warmup')
    print(json.dumps(result, indent=2))
    return result
