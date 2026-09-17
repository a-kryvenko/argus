"""Consume anonymous Nginx JSON records with transactional, restart-safe cursors.

A shared advisory lock prevents duplicate ingestion across API workers. No URL,
IP, user agent, session or account identifier enters this pipeline.
"""
import asyncio
import json
import logging
import math
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, or_, and_, text
from sqlalchemy.dialects.postgresql import insert
from app.db.models.monitoring import MonitorState, TrafficMetric
from app.db.session import get_session_factory
from app.services.api_statistics import BOUNDS, summary
from app.services.project_monitoring import save_state

logger = logging.getLogger(__name__)
MAX_BATCH_BYTES = 4 * 1024 * 1024


def read_batch(path, cursor):
    """Read complete lines only. Finish renamed file before switching to new inode."""
    current = Path(path)
    target = current
    offset = int(cursor.get('offset', 0))
    identity = cursor.get('identity')
    def file_id(stat):
        return f'{stat.st_dev}:{stat.st_ino}'
    stat = current.stat()
    if identity and identity != file_id(stat):
        # Rotation must keep uncompressed old files until the reader catches up.
        for candidate in current.parent.glob(current.name + '.*'):
            if file_id(candidate.stat()) == identity and candidate.stat().st_size > offset:
                target = candidate
                break
    with target.open('rb') as stream:
        stat = os.fstat(stream.fileno())
        if identity != file_id(stat) or offset > stat.st_size:
            offset = 0
        stream.seek(offset)
        data = stream.read(MAX_BATCH_BYTES)
        complete = data.rfind(b'\n') + 1
        if not complete and len(data) == MAX_BATCH_BYTES:
            raise ValueError('Oversized monitoring record')
        data = data[:complete]
        next_cursor = {'identity': file_id(stat), 'offset': offset+complete}
        caught_up = target == current and offset+complete >= stat.st_size
    return data.splitlines(), next_cursor, caught_up


def aggregate(lines, now):
    buckets = defaultdict(lambda: [0, 0.0])
    invalid = 0
    for line in lines:
        try:
            event = json.loads(line)
            channel = event['channel']
            if channel == 'monitor':
                continue
            status = int(event['status'])
            duration = float(event['seconds'])*1000
            stamp = datetime.fromtimestamp(float(event['time']), UTC)
            if channel not in {'site', 'api'} or not 100 <= status <= 599 or not math.isfinite(duration) or duration < 0:
                raise ValueError('Invalid record')
            if stamp > now + timedelta(minutes=5) or stamp < now - timedelta(days=30):
                continue
            bucket = next((b for b in BOUNDS if duration <= b), -1)
            for resolution in ('minute', 'hour'):
                if resolution == 'minute' and stamp < now - timedelta(hours=48):
                    continue
                when = stamp.replace(second=0, microsecond=0, **({'minute': 0} if resolution == 'hour' else {}))
                entry = buckets[(when, resolution, channel, status, bucket)]
                entry[0] += 1
                entry[1] += duration
        except (ValueError, KeyError, TypeError, OverflowError, OSError):
            invalid += 1
    return buckets, invalid


async def ingest():
    path = os.getenv('MONITORING_TRAFFIC_LOG')
    if not path:
        return
    async with get_session_factory()() as db:
        await db.execute(text("SET LOCAL statement_timeout = '10s'"))
        if not await db.scalar(text('SELECT pg_try_advisory_xact_lock(730311)')):
            return
        state = await db.get(MonitorState, 'traffic')
        now = datetime.now(UTC)
        previous = state.payload if state else {}
        if state and now - state.checked_at < timedelta(seconds=10):
            return
        try:
            lines, cursor, caught_up = await asyncio.to_thread(read_batch, path, previous)
            buckets, invalid = aggregate(lines, now)
        except (OSError, ValueError):
            await save_state(db, 'traffic', {**previous, 'status': 'unknown', 'error': 'Traffic log is unavailable'}, now)
            await db.commit()
            return
        values = [dict(time=when, resolution=resolution, channel=channel, status=status,
                       bucket_ms=bucket, count=count, duration_ms=duration)
                  for (when, resolution, channel, status, bucket), (count, duration) in buckets.items()]
        for offset in range(0, len(values), 500):
            stmt = insert(TrafficMetric).values(values[offset:offset+500])
            await db.execute(stmt.on_conflict_do_update(index_elements=['time', 'resolution', 'channel', 'status', 'bucket_ms'],
                set_={'count': TrafficMetric.count+stmt.excluded.count,
                      'duration_ms': TrafficMetric.duration_ms+stmt.excluded.duration_ms}))
        await db.execute(delete(TrafficMetric).where(or_(
            and_(TrafficMetric.resolution == 'minute', TrafficMetric.time < now-timedelta(hours=48)),
            TrafficMetric.time < now-timedelta(days=30))))
        last_event = previous.get('last_event_at')
        for line in reversed(lines):
            try:
                stamp = datetime.fromtimestamp(float(json.loads(line)['time']), UTC)
                if stamp <= now + timedelta(seconds=5):
                    last_event = stamp.isoformat()
                    break
            except (ValueError, KeyError, TypeError, OverflowError, OSError):
                pass
        await save_state(db, 'traffic', {**cursor, 'last_event_at': last_event, 'status': 'ok' if caught_up and not invalid else 'warning',
            'caught_up': caught_up, 'invalid_records': invalid,
            'since': previous.get('since', now.isoformat()), 'error': None}, now)
        await db.commit()


async def run_traffic_loop():
    while True:
        try:
            await ingest()
        except Exception:
            logger.warning('Traffic aggregation failed', exc_info=True)
        await asyncio.sleep(15)


def traffic_summary(rows, since, until, resolution):
    step = timedelta(minutes=1) if resolution == 'minute' else timedelta(hours=1)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.channel, row.time)].append(row)
    channels = {}
    for channel in ('api', 'site'):
        points = []
        stamp = since
        while stamp <= until:
            points.append({'time': stamp.isoformat(), **summary(grouped[(channel, stamp)])})
            stamp += step
        channels[channel] = {'summary': summary([r for r in rows if r.channel == channel]), 'points': points}
    return channels
