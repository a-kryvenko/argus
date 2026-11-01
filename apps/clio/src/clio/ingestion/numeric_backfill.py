"""Gap-only numeric backfill, using each observation's historical cadence."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from requests import RequestException

from clio.config import ClioObservations
from clio.ingestion.adapters import historical_adapters
from clio.ingestion.products import OBSERVATIONS
from clio.ingestion.normalize import normalize_measurements
from clio.observations.measurements import load_measurements, upsert_measurements
from clio.observations.normalized import upsert_normalized_observations

logger = logging.getLogger(__name__)

def missing_slots(stored, metric, start, end, *, freshness_minutes):
    """Raw records close their native slot; normalized rows are never read."""
    resolution = OBSERVATIONS[metric].resolution
    start, end = pd.Timestamp(start).ceil(resolution), pd.Timestamp(end).floor(resolution)
    clock = set(pd.date_range(start, end, freq=resolution, inclusive='left')) if start < end else set()
    rows = stored.loc[stored.metric == metric].copy()
    rows = rows.loc[rows.value.map(lambda value: pd.notna(value) and OBSERVATIONS[metric].accepts(value))]
    times = pd.to_datetime(rows.observed_at, utc=True)
    times = times.loc[times.ge(start) & times.lt(end)]
    if OBSERVATIONS[metric].coverage == 'latest':
        # Published background means are held between releases, not a daily
        # observation series to reconstruct through the publication delay.
        if not times.empty and end - times.max() <= timedelta(minutes=freshness_minutes):
            return set()
        return clock
    present = set(times.dt.floor(resolution))
    return clock - present


def request_ranges(hours):
    """Coalesce provider calendar days, bounded to 31 days per request."""
    days = sorted({hour.floor('D').to_pydatetime() for hour in hours})
    if not days:
        return
    start = previous = days[0]
    for day in days[1:]:
        if day != previous + timedelta(days=1) or day - start >= timedelta(days=31):
            yield start, previous + timedelta(days=1)
            start = day
        previous = day
    yield start, previous + timedelta(days=1)


def fetch_missing(pending, policies, adapters):
    """Each priority round batches metrics sharing a provider request."""
    records, attempts, cache = [], [], {}
    pending = {metric: set(hours) for metric, hours in pending.items()}
    for rank in range(max(len(policies[m].sources.historical) for m in pending)):
        products = {}
        for metric, hours in pending.items():
            sources = policies[metric].sources.historical
            if hours and rank < len(sources):
                products.setdefault(sources[rank], []).append(metric)
        for product, metrics in products.items():
            hours = set().union(*(pending[m] for m in metrics))
            for start, end in request_ranges(hours):
                if not any(start <= slot < end for metric in metrics for slot in pending[metric]):
                    continue
                attempt = {'product': product, 'metrics': metrics,
                           'from': start.isoformat(), 'to_exclusive': end.isoformat()}
                try:
                    cached = next((entry for entry in cache.get(product, [])
                                   if entry[0] <= start and entry[1] >= end), None)
                    if cached is None:
                        frame = adapters[product].fetch(start, end)
                        cache.setdefault(product, []).append((start, end, frame))
                    else:
                        frame = cached[2]
                    received_at = datetime.now(UTC)
                    accepted = 0
                    latest = frame.metric.map(lambda m: m in OBSERVATIONS and OBSERVATIONS[m].coverage == 'latest')
                    frame = pd.concat([frame.loc[~latest].sort_values('observed_at'),
                                       frame.loc[latest].sort_values('observed_at', ascending=False)])
                    for row in frame.itertuples(index=False):
                        observed = pd.Timestamp(row.observed_at)
                        if observed.tzinfo is None:
                            raise ValueError('Adapter returned a timezone-naive observation')
                        observed = observed.tz_convert('UTC')
                        if row.metric not in metrics:
                            continue
                        definition = OBSERVATIONS[row.metric]
                        slot = observed.floor(definition.resolution)
                        # Hourly products must return hourly samples. Daily and
                        # three-hour coverage keeps the original observation time
                        # (for example GFZ noon) instead of relabeling it.
                        if ((definition.resolution == '1h' and observed != slot)
                                or not start <= observed < end or slot not in pending[row.metric]
                                or not definition.accepts(row.value)):
                            continue
                        records.append({'metric': row.metric, 'value': row.value,
                                        'observed_at': observed, 'source_product': product,
                                        'received_at': received_at})
                        if definition.coverage == 'latest':
                            pending[row.metric].clear()
                        else:
                            pending[row.metric].remove(slot)
                        accepted += 1
                    attempt['accepted_measurements'] = accepted
                except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                    attempt['error'] = str(exc)
                attempts.append(attempt)
    return pd.DataFrame(records, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at']), attempts


async def backfill_numeric(session, config: ClioObservations, metrics, *, now=None, raise_on_failure=True):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(metric not in config.observations or OBSERVATIONS[metric].kind != 'numeric' for metric in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    end = now.astimezone(UTC)
    ends = {m: pd.Timestamp(end).floor(OBSERVATIONS[m].resolution).to_pydatetime() for m in metrics}
    starts = {m: ends[m] - timedelta(days=config.observations[m].backfill_days) for m in metrics}
    earliest = min(starts.values())
    latest = max(ends.values())
    stored = await load_measurements(session, since=earliest, until=latest)
    pending = {m: missing_slots(stored, m, starts[m], ends[m],
                                freshness_minutes=config.observations[m].freshness_minutes) for m in metrics}
    logger.info('Numeric backfill missing slots: %s', {
        m: min(1, len(slots)) if OBSERVATIONS[m].coverage == 'latest' else len(slots)
        for m, slots in pending.items()})
    if any(pending.values()):
        frame, attempts = await asyncio.to_thread(fetch_missing, pending, config.observations, historical_adapters())
    else:
        frame, attempts = pd.DataFrame(columns=['metric', 'value', 'observed_at']), []
    if raise_on_failure and frame.empty and any('error' in attempt for attempt in attempts):
        raise RuntimeError('No gaps filled; source errors: ' + '; '.join(
            f"{a['product']}: {a['error']}" for a in attempts if 'error' in a))
    await upsert_measurements(session, frame, replace_existing=False, repair_missing=True)
    # Reload after insert: concurrent or pre-existing observations always win.
    stored = await load_measurements(session, since=earliest, until=latest)
    normalized = normalize_measurements(stored)
    normalized = normalized.loc[normalized.observed_at >= now - timedelta(days=config.retention.normalized_days)]
    await upsert_normalized_observations(session, normalized, replace_existing=False)
    await session.commit()
    remaining = {m: (min(1, len(slots)) if OBSERVATIONS[m].coverage == 'latest' else len(slots))
                 for m in metrics for slots in [missing_slots(stored, m, starts[m], ends[m],
                                freshness_minutes=config.observations[m].freshness_minutes)]}
    failed_metrics = {m for attempt in attempts if 'error' in attempt for m in attempt['metrics'] if remaining[m]}
    return {'status': 'complete' if not any(remaining.values()) else 'partial',
            'ranges': {m: {'from': starts[m].isoformat(), 'to_exclusive': ends[m].isoformat(),
                          'resolution': OBSERVATIONS[m].resolution} for m in metrics},
            'downloaded_measurements': len(frame), 'normalized_hours': len(normalized),
            'missing_observed_slots': remaining,
            'missing_observed_hours': {m: count for m, count in remaining.items() if OBSERVATIONS[m].resolution == '1h'},
            'failed_metrics': sorted(failed_metrics), 'source_attempts': attempts}
