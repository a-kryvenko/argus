"""Archive source GOES samples without deriving solar indices."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from clio.ingestion.utils.file_observations import find, save

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


def download_snapshots(start, end, historical, expected=None, *, resolution_minutes):
    expected = expected or {}
    records = []
    source = 'goes.archive' if historical else 'goes.live'
    slots = pd.date_range(pd.Timestamp(start).ceil(f'{resolution_minutes}min'), end - timedelta(microseconds=1), freq=f'{resolution_minutes}min') if historical else [pd.Timestamp(end).floor(f'{resolution_minutes}min')]
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
        groups = frame.groupby(frame.timestamp.dt.floor('D')) if historical else [(pd.Timestamp(end).floor(f'{resolution_minutes}min'), frame)]
        for slot, samples in groups:
            slot = slot.to_pydatetime()
            if samples.empty or slot not in pending:
                continue
            with TemporaryDirectory(prefix='clio-goes-') as temporary:
                path = Path(temporary) / 'original.json'
                samples.to_json(path, orient='records', date_format='iso')
                records.append(save(path, observation='goes', observed_at=samples.timestamp.max().to_pydatetime(),
                                    source_product=source))
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
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now:
        raise ValueError('Choose a past timezone-aware GOES range')
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if historical and any(t.hour or t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError('GOES archive ranges must use whole UTC days')
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    try:
        rows = await asyncio.to_thread(find, 'goes', start=start, end=end+timedelta(microseconds=1))
        resolution = policy.resolution_minutes if historical else policy.live_resolution_minutes
        expected = {(pd.Timestamp(row['observed_at']).floor(f'{resolution}min').to_pydatetime(), row['source_product']): row
                    for row in rows}
        records = await asyncio.to_thread(download_snapshots, start, end, historical, expected,
            resolution_minutes=resolution)
        saved_ids = {row['id'] for row in rows}
        report['retained'] = sum(record['id'] in saved_ids for record in records)
        report['received'] = len(records)-report['retained']
        latest = max((r['observed_at'] for r in records), default=None)
        report['latest_observed_at'] = latest.isoformat() if latest else None
        report['missing'] = max(0, (end.date()-start.date()).days-len(records)) if historical else int(latest is None or latest < now-timedelta(minutes=policy.freshness_minutes))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': 'goes.archive' if historical else 'goes.live', 'error': str(exc)})
    failed = bool(report['failed'] or report['missing'])
    return {'status': 'partial' if failed else 'complete', 'failed_metrics': ['goes'] if failed else [], 'files': {'goes': report}}
