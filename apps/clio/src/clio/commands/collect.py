"""Fetch configured numeric/file observations without running normalization."""
import json

from clio.db.session import get_session_factory
from clio.config import load_observation_config
from clio.ingestion.selection import partition, merge_files, available_files


async def run(args):
    config = load_observation_config()
    numeric, files = partition(config, args.metrics)
    result = {'status': 'complete', 'failed_metrics': [], 'downloaded_measurements': 0}
    if numeric:
        from clio.observations.live import collect_live
        async with get_session_factory()() as session:
            result = await collect_live(session, config, numeric,
                                        now=getattr(args, 'now', None), heartbeat=getattr(args, 'heartbeat', None))
    if files:
        from clio.domains.aia.collection import collect_file_observations
        result = merge_files(result, await collect_file_observations(config, files, mode='live', now=getattr(args, 'now', None)))
    print(json.dumps(result, indent=2))
    if (not getattr(args, 'scheduled', False) and result['downloaded_measurements'] == 0
            and result['failed_metrics'] and not available_files(result)):
        raise RuntimeError('No live observations available: ' + ', '.join(result['failed_metrics']))
    return result
