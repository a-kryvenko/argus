"""Download immutable GONG originals and retain their first receipts."""
import asyncio
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.observations.files import observation_values, snapshot_receipt


def snapshot_values(observed, url, source, *, expected=None):
    from clio.providers import gong as provider
    from clio.observations.files import store_original
    slot = observed.replace(minute=0, second=0, microsecond=0)
    values = store_original('gong', slot, observed, source, lambda: provider.download(url), expected=expected, metadata={'source_url': url})
    return {**values, 'source_url': values.get('source_url', url)}


async def collect(config, *, mode, now=None, start=None, end=None):
    from clio.providers import gong as provider
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    policy = config.observations['gong']
    historical = mode == 'backfill'
    if mode not in ('live', 'backfill') or (start is None) != (end is None):
        raise ValueError('Use live/backfill and both range boundaries')
    start = (start or now - timedelta(days=policy.backfill.days)) if historical else now - timedelta(hours=3)
    end = end or now
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now or end-start > timedelta(days=60):
        raise ValueError('Choose a past aware GONG range up to 60 days')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    source = (policy.sources.historical if historical else policy.sources.live)[0]
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    async with get_db_session() as session:
        records = (await session.scalars(select(FileObservation).where(
            FileObservation.kind == 'gong',
            FileObservation.source_key >= start.replace(minute=0, second=0, microsecond=0).isoformat(),
            FileObservation.source_key < (end + timedelta(hours=1)).isoformat()))).all()
    records = await asyncio.to_thread(lambda: [snapshot_receipt(row) for row in records])
    records = [row for row in records if start <= row['observed_at'] <= end]
    existing = {row['observed_at'] for row in records}
    usable = set()
    from clio.observations.files import archive_root, store_original
    for row in records:
        try:
            present = (archive_root('gong') / row['raw_path']).is_file()
            await asyncio.to_thread(store_original, 'gong', row['slot_at'], row['observed_at'],
                row['source_product'], lambda: provider.download(row['source_url']),
                expected=row, relative=row['raw_path'])
            report['restored' if not present else 'retained'] += 1
            usable.add(row['observed_at'])
        except Exception as exc:
            report['failed'] += 1
            if len(report['source_errors']) < 20:
                report['source_errors'].append({'product': source, 'error': str(exc)})
    latest = max(usable) if usable else None
    report['retained'] = len(usable) - report['restored']
    try:
        candidates = await asyncio.to_thread(provider.candidates, start, end, historical=historical)
        # One magnetogram per observed hour; keep already received hours immutable.
        hours = {t.replace(minute=0, second=0, microsecond=0) for t in existing}
        for observed, url in candidates:
            hour = observed.replace(minute=0, second=0, microsecond=0)
            if hour in hours:
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
                if inserted:
                    latest = max(latest, observed) if latest else observed
            except Exception as exc:
                report['failed'] += 1
                if len(report['source_errors']) < 20:
                    report['source_errors'].append({'product': source, 'error': str(exc)})
        expected = max(1, int((end-start).total_seconds() // 3600))
        report['missing'] = max(0, expected-len(hours))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': source, 'error': str(exc)})
    report['latest_observed_at'] = latest.isoformat() if latest else None
    failed = report['failed'] > 0 if historical else latest is None or latest < now-timedelta(hours=3)
    return {'status': 'partial' if failed or report['missing'] or report['failed'] else 'complete',
            'failed_metrics': ['gong'] if failed else [], 'files': {'gong': report}}
