"""Download GONG originals to tmp and save uploaded observations."""
import asyncio
import math
from datetime import UTC, datetime, timedelta

from pathlib import Path
from tempfile import TemporaryDirectory

from clio.ingestion.utils.file_observations import find, save


def snapshot_values(observed, url, source):
    from clio.providers import gong as provider
    with TemporaryDirectory(prefix='clio-gong-') as temporary:
        path = Path(temporary) / 'original.fits'
        path.write_bytes(provider.download(url))
        return save(path, observation='gong', observed_at=observed,
                    source_product=source, details={'source_url': url})


async def collect(config, *, mode, now=None, start=None, end=None):
    from clio.db.locks import ingestion_lock, FILE_LOCKS
    with ingestion_lock('gong', FILE_LOCKS['gong']) as acquired:
        if not acquired:
            return {'status': 'busy', 'files': {'gong': {'status': 'busy'}}, 'failed_metrics': []}
        return await _collect(config, mode=mode, now=now, start=start, end=end)


async def _collect(config, *, mode, now=None, start=None, end=None):
    from clio.providers import gong as provider
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.astimezone(UTC)
    policy = config.observations['gong']
    historical = mode == 'backfill'
    if mode not in ('live', 'backfill') or (start is None) != (end is None):
        raise ValueError('Use live/backfill and both range boundaries')
    end = end or (now.replace(minute=0, second=0, microsecond=0) if historical else now)
    start = (start or end - timedelta(days=policy.backfill_days)) if historical else now - timedelta(minutes=policy.freshness_minutes)
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now:
        raise ValueError('Choose a past timezone-aware GONG range')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    source = (policy.sources.historical if historical else policy.sources.live)[0]
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    records = await asyncio.to_thread(find, 'gong', start=start, end=end+timedelta(microseconds=1))
    existing = {row['observed_at'] for row in records}
    usable = set(existing)
    latest = max(usable) if usable else None
    report['retained'] = len(usable)
    try:
        # One magnetogram per closed hour; retain existing database references.
        hours = {t.replace(minute=0, second=0, microsecond=0) for t in existing}
        required = set()
        if historical:
            step = policy.resolution_minutes * 60
            hour = datetime.fromtimestamp(math.ceil(start.timestamp() / step) * step, UTC)
            while hour < end.replace(minute=0, second=0, microsecond=0):
                required.add(hour)
                hour += timedelta(minutes=policy.resolution_minutes)
            pending = required - hours
            candidates = []
            for day in sorted({hour.replace(hour=0) for hour in pending}):
                frame = await asyncio.to_thread(provider.candidates, max(start, day),
                    min(end, day + timedelta(days=1)), historical=True)
                candidates.extend(frame.itertuples(index=False, name=None))
        else:
            frame = await asyncio.to_thread(provider.candidates, start, end, historical=False)
            candidates = frame.itertuples(index=False, name=None)
        for observed, url in candidates:
            hour = observed.replace(minute=0, second=0, microsecond=0)
            resolution = policy.resolution_minutes if historical else policy.live_resolution_minutes
            if int(hour.timestamp()) % (resolution * 60) or hour in hours or (historical and hour not in required):
                continue
            try:
                values = await asyncio.to_thread(snapshot_values, observed, url, source)
                report['received'] += 1
                hours.add(hour)
                usable.add(observed)
                if inserted:
                    latest = max(latest, observed) if latest else observed
            except Exception as exc:
                report['failed'] += 1
                if len(report['source_errors']) < 20:
                    report['source_errors'].append({'product': source, 'error': str(exc)})
        if historical:
            report['missing'] = len(required - {t.replace(minute=0, second=0, microsecond=0) for t in usable})
        else:
            expected = max(1, int((end-start).total_seconds() // (policy.live_resolution_minutes * 60)))
            report['missing'] = max(0, expected-len(hours))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': source, 'error': str(exc)})
    report['latest_observed_at'] = latest.isoformat() if latest else None
    failed = report['failed'] > 0 if historical else latest is None or latest < now-timedelta(minutes=policy.freshness_minutes)
    return {'status': 'partial' if failed or report['missing'] or report['failed'] else 'complete',
            'failed_metrics': ['gong'] if failed else [], 'files': {'gong': report}}
