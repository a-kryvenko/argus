"""Read saved complete aggregate windows without fetching or recalculating data."""
from datetime import UTC, datetime, timedelta
import math
from sqlalchemy import select
from argus_clio.db.models import SolarWindAggregate, SolarWindAggregatePending
from argus_clio.services.solar_wind.observations import METADATA, metadata
from argus_clio.services.solar_wind.aggregation import VERSION


async def history(session, metrics, start, end, seconds, now=None):
    now = now or datetime.now(UTC)
    left = datetime.fromtimestamp(math.floor(start.timestamp()/seconds)*seconds, UTC)
    right = datetime.fromtimestamp(math.floor(min(end, now).timestamp()/seconds)*seconds, UTC)
    kinds = {METADATA[m][0] for m in metrics}
    records = list((await session.execute(select(SolarWindAggregate).where(
        SolarWindAggregate.kind.in_(kinds), SolarWindAggregate.resolution_seconds == seconds,
        SolarWindAggregate.bucket_start >= left, SolarWindAggregate.bucket_start < right,
        SolarWindAggregate.version == VERSION).order_by(SolarWindAggregate.bucket_start))).scalars())
    pending = {(r.kind, r.hour) for r in (await session.execute(select(SolarWindAggregatePending).where(
        SolarWindAggregatePending.kind.in_(kinds),
        SolarWindAggregatePending.hour >= left.replace(minute=0),
        SolarWindAggregatePending.hour < right))).scalars()}
    by_key = {(r.kind, r.bucket_start): r for r in records}
    series = {}
    for metric in metrics:
        kind = METADATA[metric][0]
        points = []
        valid = invalid = missing = unavailable = recalculating = 0
        gaps = []
        cursor = left
        while cursor < right:
            row = by_key.get((kind, cursor))
            ready = row is not None and row.statistics['window_complete']
            waiting = (kind, cursor.replace(minute=0)) in pending
            stats = row.statistics['metrics'][metric] if ready else None
            count = stats['count'] if stats else 0
            valid += count
            invalid += stats['invalid_count'] if stats else 0
            missing += stats['missing_count'] if stats else 0
            unavailable += int(not ready)
            recalculating += int(ready and waiting)
            if stats and count:
                sequence = row.statistics['source_sequence']
                points.append({'observed_at': cursor, 'interval_end': cursor+timedelta(seconds=seconds),
                    'received_at': row.calculated_at, 'value': stats['mean'], 'min': stats['min'], 'max': stats['max'],
                    'count': count, 'expected_count': stats['expected_count'], 'coverage_percent': stats['coverage_percent'],
                    'recalculation_pending': waiting,
                    'spacecraft': sequence[0]['spacecraft'] if sequence else '',
                    'last_spacecraft': sequence[-1]['spacecraft'] if sequence else '',
                    'source_changes': row.statistics['source_changes'],
                    'quality': 'unverified', 'provider_quality': None,
                    'negative_count': stats.get('negative_count')})
            if stats and count < seconds//60:
                reason = 'partial'
                gap_end = cursor+timedelta(seconds=seconds)
                if gaps and gaps[-1]['to'] == cursor and gaps[-1]['reason'] == reason:
                    gaps[-1]['to'] = gap_end
                    gaps[-1]['slots'] += seconds//60-count
                else:
                    gaps.append({'from': cursor, 'to': gap_end, 'reason': reason, 'slots': seconds//60-count})
            cursor += timedelta(seconds=seconds)
        expected = valid+missing+invalid
        series[metric] = {**metadata(metric), 'resolution_seconds': seconds, 'aggregation': 'mean_min_max',
            'processing': {'unavailable_buckets': unavailable, 'recalculation_pending_buckets': recalculating,
                           'available_buckets': expected//(seconds//60),
                           'expected_buckets': expected//(seconds//60)+unavailable},
            'points': points, 'coverage': {'expected_slots': expected, 'usable_slots': valid,
            'missing_slots': missing, 'invalid_slots': invalid, 'percent': round(100*valid/expected, 2) if expected else None,
            'resolution_seconds': 60, 'evaluated_to': right, 'basis': 'available_aggregate_windows',
            'gaps': gaps}}
    return {'from': start, 'to': end, 'evaluated_from': left, 'evaluated_to': right,
            'resolution_seconds': seconds, 'aggregation_version': VERSION,
            'selection': 'complete_UTC_buckets_starting_before_to_left_edge_may_extend_before_from',
            'gap_filling': 'none', 'series': series}
