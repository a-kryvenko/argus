"""Archive source GOES samples without deriving solar indices."""
import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.goes_snapshot import GOESSnapshot
from clio.db.session import get_session_factory
from clio.observations.files import archive_root, store_original


def source_frames(start, end, historical):
    if not historical:
        from clio.providers.goes_loader import fetch_goes
        yield fetch_goes(end=end, timeout=25)
        return
    from clio.providers.goes_history_loader import archive_session, discover_goes_history, iter_goes_history
    satellites = [int(value) for value in os.getenv('GOES_ARCHIVE_SATELLITES', '18,16').split(',')]
    days = pd.date_range(start, end - timedelta(microseconds=1), freq='D')
    with archive_session() as session:
        files = discover_goes_history(sorted(set(days.year)), satellites, session)
        for _, frame in iter_goes_history(files, days, satellites, archive_root('goes') / 'source', session, refresh=True):
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
        if saved and (archive_root('goes') / saved['raw_path']).is_file():
            records.append(store_original('goes', slot, saved['observed_at'], source,
                lambda: None, relative=Path(saved['raw_path']), expected=saved))
        else:
            pending.add(slot)
    if not pending:
        return records
    for frame in source_frames(start, end, historical):
        frame = frame.loc[frame.timestamp.le(end)].copy()
        if historical:
            frame = frame.loc[frame.timestamp.ge(start) & frame.timestamp.lt(end)]
        groups = frame.groupby(frame.timestamp.dt.floor('D')) if historical else [(pd.Timestamp(end).floor('h'), frame)]
        for slot, samples in groups:
            slot = slot.to_pydatetime()
            if samples.empty or slot not in pending:
                continue
            content = samples.to_json(orient='records', date_format='iso').encode()
            values = store_original('goes', slot, samples.timestamp.max().to_pydatetime(), source, lambda: content,
                relative=Path(source) / f'{slot:%Y/%m/%d/%H}.json', expected=expected.get((slot, source)))
            records.append(values)
    return records


async def collect(config, *, mode, now=None, start=None, end=None):
    now = now or datetime.now(UTC)
    if mode not in ('live', 'backfill') or now.tzinfo is None or (start is None) != (end is None):
        raise ValueError('Use live/backfill and aware range boundaries')
    historical = mode == 'backfill'
    end = end or (now.replace(hour=0, minute=0, second=0, microsecond=0) if historical else now)
    start = start or end - timedelta(days=config.observations['goes'].backfill.days if historical else 1)
    if start.tzinfo is None or end.tzinfo is None or not start < end <= now or end - start > timedelta(days=90):
        raise ValueError('Choose a past GOES range up to 90 days')
    if historical and any(t.hour or t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError('GOES archive ranges must use whole UTC days')
    report = dict(received=0, restored=0, retained=0, missing=0, rejected=0, failed=0,
                  latest_observed_at=None, source_errors=[])
    try:
        async with get_session_factory()() as session:
            rows = (await session.scalars(select(GOESSnapshot).where(
                GOESSnapshot.slot_at >= start, GOESSnapshot.slot_at <= end))).all()
        expected = {(row.slot_at, row.source_product): {key: getattr(row, key) for key in (
            'slot_at', 'observed_at', 'available_at', 'sha256', 'source_product', 'raw_path')} for row in rows}
        records = await asyncio.to_thread(download_snapshots, start, end, historical, expected)
        async with get_session_factory()() as session:
            for record in records:
                inserted = (await session.execute(insert(GOESSnapshot).values(**record)
                    .on_conflict_do_nothing(index_elements=['slot_at', 'source_product']).returning(GOESSnapshot.slot_at))).scalar_one_or_none()
                report['received' if inserted else 'retained'] += 1
            await session.commit()
        latest = max((r['observed_at'] for r in records), default=None)
        report['latest_observed_at'] = latest.isoformat() if latest else None
        report['missing'] = max(0, (end.date()-start.date()).days-len(records)) if historical else int(latest is None or latest < now-timedelta(hours=2))
    except Exception as exc:
        report['failed'] += 1
        report['source_errors'].append({'product': 'goes.archive' if historical else 'goes.live', 'error': str(exc)})
    failed = bool(report['failed'] or report['missing'])
    return {'status': 'partial' if failed else 'complete', 'failed_metrics': ['goes'] if failed else [], 'files': {'goes': report}}
