"""Dispatch file products and report source freshness per observation."""
import asyncio
from datetime import datetime, timedelta
from importlib import import_module

from clio.ingestion.utils.products import OBSERVATIONS
from clio.ingestion.utils.file_observations import find


async def collect_observation(config, observation, *, mode, now):
    definition = OBSERVATIONS[observation]
    policy = config.observations[observation]
    sources = policy.sources.live if mode == 'live' else policy.sources.historical
    if not sources:
        return {'status': 'skipped', 'failed': 0}
    handler = definition.handler or observation
    module = import_module('clio.ingestion.extras.' + ('sdo' if handler == 'sdo_nrt' else handler))
    if definition.channel is None:
        result = await module.collect(config, mode=mode, now=now)
        return dict(result['files'][observation], status=result['status'])
    for source in sources:
        report = await asyncio.to_thread(module.collect, policy, channel=definition.channel,
            source=source, mode=mode, now=now)
        if report['status'] == 'busy':
            return report
        if report['status'] == 'complete':
            break
    rows = await asyncio.to_thread(find, observation)
    observed = []
    for row in rows:
        if row['source_product'] not in sources:
            continue
        metadata = row['details'] or {}
        value = row['observed_at'] if handler == 'sdo_nrt' else metadata.get('source_updated_at')
        if value is not None:
            timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
            if timestamp.tzinfo is not None and timestamp <= now:
                observed.append(timestamp)
    latest = max(observed, default=None)
    report['latest_observed_at'] = latest.isoformat() if latest else None
    report['freshness'] = ('unknown' if latest is None else 'fresh'
        if latest >= now - timedelta(minutes=policy.freshness_minutes) else 'stale')
    if mode == 'live' and report['freshness'] != 'fresh':
        report['status'] = 'partial'
    return report
