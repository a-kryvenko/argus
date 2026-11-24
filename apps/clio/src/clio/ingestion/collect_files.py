"""Run file collectors and summarize their outcomes."""
import asyncio
from datetime import datetime, timedelta
from functools import partial

from clio.ingestion.utils.file_observations import find
from clio.ingestion.utils.products import OBSERVATIONS


async def collect_files(config, observations, *, mode, now):
    selected = list(dict.fromkeys(observations))
    if any(observation not in config.observations or OBSERVATIONS[observation].kind != 'file'
           for observation in selected):
        raise ValueError('Select configured file observations')
    files = {}
    for observation in selected:
        files[observation] = await collect_observation(config, observation, mode=mode, now=now)

    failed = [observation for observation, report in files.items() if report['status'] == 'partial']
    if failed:
        status = 'partial'
    elif any(report['status'] == 'busy' for report in files.values()):
        status = 'busy'
    elif files and all(report['status'] == 'skipped' for report in files.values()):
        status = 'skipped'
    else:
        status = 'complete'
    return {'status': status, 'failed_metrics': sorted(failed), 'files': files}


async def collect_observation(config, observation, *, mode, now):
    from clio.ingestion.extras import goes, gong, sdo, sdo_browse

    if mode not in ('live', 'backfill'):
        raise ValueError('Use live/backfill')
    definition = OBSERVATIONS[observation]
    policy = config.observations[observation]
    sources = policy.sources.live if mode == 'live' else policy.sources.historical
    if not sources:
        return {'status': 'skipped', 'failed': 0}

    kind = definition.file_kind or observation
    handlers = {
        'goes': goes.collect,
        'gong': gong.collect,
        'sdo': partial(asyncio.to_thread, sdo.collect, channel=definition.channel),
        'sdo_browse': partial(asyncio.to_thread, sdo_browse.collect, channel=definition.channel),
    }
    collect = handlers[kind]
    for source in sources:
        report = await collect(policy, source=source, mode=mode, now=now)
        if report['status'] in ('busy', 'complete'):
            break

    if report['status'] != 'busy' and kind in ('sdo', 'sdo_browse'):
        await _set_sdo_freshness(observation, policy, sources, report, browse=kind == 'sdo_browse',
                                 mode=mode, now=now)
    return report


async def _set_sdo_freshness(observation, policy, sources, report, *, browse, mode, now):
    rows = await asyncio.to_thread(find, observation)
    latest = None
    for row in rows:
        if row['source_product'] not in sources:
            continue
        metadata = row['details'] or {}
        value = metadata.get('source_updated_at') if browse else row['observed_at']
        if value is None:
            continue
        timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
        if timestamp.tzinfo is not None and timestamp <= now and (latest is None or timestamp > latest):
            latest = timestamp

    report['latest_observed_at'] = latest.isoformat() if latest else None
    if latest is None:
        report['freshness'] = 'unknown'
    elif latest >= now - timedelta(minutes=policy.freshness_minutes):
        report['freshness'] = 'fresh'
    else:
        report['freshness'] = 'stale'
    if mode == 'live' and report['freshness'] != 'fresh':
        report['status'] = 'partial'
