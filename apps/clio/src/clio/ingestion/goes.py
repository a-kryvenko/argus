"""Archive source GOES samples without deriving solar indices."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.file_observation import FileObservation
from clio.db.session import get_db_session
from clio.observations.archive import observation_values
from clio.observations.file_catalog import snapshots
from clio.observations.archive import store_original

logger = logging.getLogger(__name__)


def source_frames(start, end, historical, *, days=None):
    if not historical:
        from clio.providers.goes import fetch_goes
        yield fetch_goes(end=end, timeout=25)
        return
    from clio.providers.goes_history import archive_session, discover_goes_history, iter_goes_history
    satellites = [int(value) for value in os.getenv('GOES_ARCHIVE_SATELLITES', '18,16').split(',')]
    days = days if days is not None else pd.date_range(start, end - timedelta(microseconds=1), freq='D')
    with archive_session() as session, TemporaryDirectory(prefix='clio-goes-') as temporary:
        files = discover_goes_history(sorted(set(days.year)), satellites, session)
        for frame in iter_goes_history(files, days, satellites, Path(temporary), session, refresh=True):
            yield frame


def download_snapshots(start, end, historical, expected=None):
    expected = expected or {}
    records = []
    source = 'goes.archive' if historical else 'goes.live'
    slots = pd.date_range(start, end - timedelta(microseconds=1), freq='D') if historical else [pd.Timestamp(end).floor('h')]
    pending = set()
    for slot in slots:
        slot = slot.to_pydatetime()
        saved = expected.get((slot, source))
        if saved is not None:
            records.append(saved)
        else:
            pending.add(slot)
    logger.info('GOES %s: retained=%d, missing=%d', 'backfill' if historical else 'live', len(records), len(pending))
    if not pending:
        return records
    options = {'days': pd.DatetimeIndex(sorted(pending))} if historical else {}
    for frame in source_frames(start, end, historical, **options):
        frame = frame.loc[frame.timestamp.le(end)].copy()
        if historical:
            frame = frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(end)]
        groups = frame.groupby(frame.timestamp.dt.floor('D')) if historical else [(pd.Timestamp(end).floor('h'), frame)]
        for slot, samples in groups:
            slot = slot.to_pydatetime()
            if samples.empty or slot not in pending:
                continue
            content = samples.to_json(orient='records', date_format='iso').encode()
            values = store_original('goes', slot, samples.timestamp.max().to_pydatetime(), source, lambda: content)
            records.append(values)
    return records


async def collect(config, *, mode, now=None, start=None, end=None):
    from clio.db.locks import ingestion_lock, FILE_LOCKS
    with ingestion_lock('goes', FILE_LOCKS['goes']) as acquired:
        if not acquired:
            return {'status': 'busy', 'files': {'goes': {'status': 'busy'}}, 'failed_metrics': []}
        return await _collect(config, mode=mode, now=now, start=start, end=end)


async def _collect(config, *, mode, now=None, start=None, end=None):
    now = now or datetime.now(UTC)
    if mode not in ('live', 'backfill') or now.tzinfo is None or (start is None) != (end is None):
        raise ValueError('Use live/backfill and aware range boundaries')
    now = now.astimezone(UTC)
    policy = config.observations['goes']
    historical = mode == 'backfill'
    end = end or (now.replace(hour=0, minute=0, second=0, microsecond=0) if historical else now)
    start = start or end - timedelta(days=policy.backfill_days if historical else 1)
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now or end - start > timedelta(days=90):
        raise ValueError('Choose a past GOES range up to 90 days')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if historical and any(t.hour or t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError('GOES archive ranges must use whole UTC days')
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    try:
        async with get_db_session() as session:
            rows = await snapshots(session, 'goes', start=start, end=end+timedelta(microseconds=1))
        expected = {(row.slot_at, row.source_product): {key: getattr(row, key) for key in (
            'slot_at', 'observed_at', 'available_at', 'sha256', 'source_product', 'file_id')} for row in rows}
        records = await asyncio.to_thread(download_snapshots, start, end, historical, expected)
        async with get_db_session() as session:
            for record in records:
                inserted = (await session.execute(insert(FileObservation).values(**observation_values('goes', record))
                    .on_conflict_do_nothing(index_elements=['kind', 'source_key'], index_where=FileObservation.error.is_(None)).returning(FileObservation.id))).scalar_one_or_none()
                report['received' if inserted else 'retained'] += 1
            await session.commit()
        latest = max((r['observed_at'] for r in records), default=None)
        report['latest_observed_at'] = latest.isoformat() if latest else None
        report['missing'] = max(0, (end.date()-start.date()).days-len(records)) if historical else int(latest is None or latest < now-timedelta(minutes=policy.freshness_minutes))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': 'goes.archive' if historical else 'goes.live', 'error': str(exc)})
    failed = bool(report['failed'] or report['missing'])
    return {'status': 'partial' if failed else 'complete', 'failed_metrics': ['goes'] if failed else [], 'files': {'goes': report}}
