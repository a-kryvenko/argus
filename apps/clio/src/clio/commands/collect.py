"""Fetch configured numeric/file observations without running normalization."""
import json

from clio.db.session import get_db_session
from clio.config import load_observation_config
from clio.ingestion.selection import partition, merge_files, merge_sdo


async def run(args):
    config = load_observation_config()
    numeric, files = partition(config, args.metrics)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.observations.live import collect_live
        async with get_db_session() as session:
            result = await collect_live(session, config, numeric,
                                        now=getattr(args, 'now', None))
    if files:
        from clio.ingestion.files import collect_file_observations
        result = merge_files(result, await collect_file_observations(config, files, mode='live', now=getattr(args, 'now', None)))
    result = await merge_sdo(result, config, args, mode='live')
    print(json.dumps(result, indent=2))
    return result
