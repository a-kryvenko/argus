"""Download immutable GONG originals and retain their first receipts."""
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy.dialects.postgresql import insert

from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.observations.archive import observation_values
from clio.observations.file_catalog import snapshots


def snapshot_values(observed, url, source):
    from clio.providers import gong as provider
    from clio.observations.archive import store_original
    slot = observed.replace(minute=0, second=0, microsecond=0)
    values = store_original('gong', slot, observed, source, lambda: provider.download(url), metadata={'source_url': url})
    return {**values, 'source_url': values.get('source_url', url)}


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
    start = (start or end - timedelta(days=policy.backfill_days)) if historical else now - timedelta(hours=3)
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now or end-start > timedelta(days=60):
        raise ValueError('Choose a past aware GONG range up to 60 days')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    source = (policy.sources.historical if historical else policy.sources.live)[0]
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    async with get_db_session() as session:
        records = await snapshots(session, 'gong', start=start, end=end+timedelta(microseconds=1))
    existing = {row.observed_at for row in records}
    usable = set(existing)
    latest = max(usable) if usable else None
    report['retained'] = len(usable)
    try:
        # One magnetogram per closed hour; retain existing database references.
        hours = {t.replace(minute=0, second=0, microsecond=0) for t in existing}
        required = set()
        if historical:
            hour = start.replace(minute=0, second=0, microsecond=0)
            if hour < start:
                hour += timedelta(hours=1)
            while hour < end.replace(minute=0, second=0, microsecond=0):
                required.add(hour)
                hour += timedelta(hours=1)
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
            if hour in hours or (historical and hour not in required):
                continue
            try:
                values = await asyncio.to_thread(snapshot_values, observed, url, source)
                async with get_db_session() as session:
                    statement = insert(FileObservation).values(**observation_values('gong', values)).on_conflict_do_nothing(
                        index_elements=['kind', 'source_key'], index_where=FileObservation.error.is_(None)).returning(FileObservation.id)
                    inserted = (await session.execute(statement)).scalar_one_or_none()
                    await session.commit()
                report['received' if inserted else 'retained'] += 1
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
            expected = max(1, int((end-start).total_seconds() // 3600))
            report['missing'] = max(0, expected-len(hours))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': source, 'error': str(exc)})
    report['latest_observed_at'] = latest.isoformat() if latest else None
    failed = report['failed'] > 0 if historical else latest is None or latest < now-timedelta(minutes=policy.freshness_minutes)
    return {'status': 'partial' if failed or report['missing'] or report['failed'] else 'complete',
            'failed_metrics': ['gong'] if failed else [], 'files': {'gong': report}}
