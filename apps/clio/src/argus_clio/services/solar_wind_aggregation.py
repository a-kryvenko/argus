"""Idempotent aggregates from active native observations; no raw data deletion."""
from datetime import UTC, datetime, timedelta
from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import insert
from argus_clio.db.models import SolarWindObservation, SolarWindAggregate, SolarWindAggregatePending
from argus_clio.db.session import get_session_factory
from clio.dataloaders.solar_wind_loader import FIELDS

VERSION = 1


def summarize(records, kind, start, seconds, now):
    end = start + timedelta(seconds=seconds)
    if end > now:
        raise ValueError('Only closed windows can be aggregated')
    # Same timestamp tie-breaking as history: newest receipt then spacecraft.
    # If a source supplies multiple timestamps in a minute, select the latest.
    selected = {}
    for row in sorted(sorted(records, key=lambda r: r.spacecraft, reverse=True),
                      key=lambda r: (r.observed_at, r.received_at)):
        if row.active and start <= row.observed_at < min(end, now):
            selected[int(row.observed_at.timestamp())//60] = row
    rows = [selected[key] for key in sorted(selected)]
    stats = {}
    for metric in FIELDS[kind]:
        valid = [r for r in rows if r.values.get(metric) is not None and r.raw.get('overall_quality') in (None, 0)]
        values = [r.values[metric] for r in valid]
        stats[metric] = {
            'count': len(valid), 'expected_count': seconds//60,
            'missing_count': seconds//60-len(rows), 'invalid_count': len(rows)-len(valid),
            'coverage_percent': round(100*len(valid)/(seconds//60), 2),
            'sum': sum(values), 'mean': sum(values)/len(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None,
            'first': {'value': values[0], 'at': valid[0].observed_at.isoformat()} if valid else None,
            'last': {'value': values[-1], 'at': valid[-1].observed_at.isoformat()} if valid else None,
        }
        if metric == 'bz':
            stats[metric]['negative_count'] = sum(value < 0 for value in values)
    return {'metrics': stats, 'source_sequence': [
        {'at': row.observed_at.isoformat(), 'spacecraft': row.spacecraft}
        for index, row in enumerate(rows) if index == 0 or rows[index-1].spacecraft != row.spacecraft],
        'source_changes': sum(a.spacecraft != b.spacecraft for a, b in zip(rows, rows[1:])),
        'max_gap_minutes': max((b-a-1 for a, b in zip(sorted(selected), sorted(selected)[1:])), default=0),
        'window_complete': True, 'evaluated_through': end.isoformat()}


async def aggregate_pending(limit=240, now=None):
    now = now or datetime.now(UTC)
    visited = []
    processed = 0
    for _ in range(limit):
        async with get_session_factory()() as session:
            # Process each source hour once per run. Keep the current hour queued
            # until it closes, without repeatedly selecting it or scanning aggregates.
            pending = (await session.execute(select(SolarWindAggregatePending)
                .where(SolarWindAggregatePending.hour < now,
                       tuple_(SolarWindAggregatePending.kind, SolarWindAggregatePending.hour).not_in(visited))
                .order_by(SolarWindAggregatePending.hour, SolarWindAggregatePending.kind)
                .with_for_update(skip_locked=True).limit(1))).scalars().first()
            if pending is None:
                break
            hour = pending.hour
            visited.append((pending.kind, hour))
            records = list((await session.execute(select(SolarWindObservation).where(
                SolarWindObservation.kind == pending.kind,
                SolarWindObservation.observed_at >= hour,
                SolarWindObservation.observed_at < hour+timedelta(hours=1)))).scalars())
            for seconds in (300, 3600):
                for offset in range(0, 3600, seconds):
                    start = hour+timedelta(seconds=offset)
                    if start+timedelta(seconds=seconds) > now:
                        continue
                    statistics = summarize(records, pending.kind, start, seconds, now)
                    statement = insert(SolarWindAggregate).values(kind=pending.kind,
                        resolution_seconds=seconds, bucket_start=start, version=VERSION,
                        calculated_at=now, statistics=statistics)
                    await session.execute(statement.on_conflict_do_update(
                        index_elements=['kind', 'resolution_seconds', 'bucket_start'],
                        set_={key: statement.excluded[key] for key in ('version', 'calculated_at', 'statistics')},
                        where=SolarWindAggregate.statistics.is_distinct_from(statement.excluded.statistics)
                              | (SolarWindAggregate.version != VERSION)))
            if hour+timedelta(hours=1) <= now:
                await session.delete(pending)
            await session.commit()
            processed += 1
    return processed
