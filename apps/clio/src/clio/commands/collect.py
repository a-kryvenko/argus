"""Fetch configured numeric/file observations without running normalization."""
import json
import logging

from clio.db.session import get_db_session
from clio.config import load_observation_config
from clio.ingestion.selection import partition, merge_files, merge_sdo

logger = logging.getLogger(__name__)


async def run(args):
    config = load_observation_config()
    catalogs = ('donki', 'enlil')
    selected = catalogs if args.metrics is None else [kind for kind in catalogs if kind in args.metrics]
    metrics = None if args.metrics is None else [metric for metric in args.metrics if metric not in catalogs]
    numeric, files = partition(config, metrics)
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
    for kind in selected:
        from clio.domains import donki, enlil
        try:
            if kind == 'donki':
                await donki.collect(config.solar_wind_events, now=getattr(args, 'now', None))
            else:
                await enlil.collect(now=getattr(args, 'now', None))
        except RuntimeError as exc:
            logger.warning('%s collection failed: %s', kind, exc)
            result[kind] = {'status': 'failed'}
            result['status'] = 'partial'
            result['failed_metrics'] = sorted(set(result['failed_metrics']) | {kind})
        else:
            result[kind] = {'status': 'complete'}
    print(json.dumps(result, indent=2))
    return result
