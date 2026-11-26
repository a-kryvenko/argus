"""Gap-only numeric backfill, using each observation's historical cadence."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
from requests import RequestException

from clio.config import ClioObservations
from clio.ingestion.utils.adapters import historical_adapters
from clio.ingestion.utils.products import OBSERVATIONS
from clio.ingestion.utils.measurements import WritePolicy, historical_records, load_measurements, write_measurements

logger = logging.getLogger(__name__)

def missing_slots(stored, metric, start, end, *, freshness_minutes, resolution_minutes, sources=None):
    """Raw records close their native slot; normalized rows are never read."""
    resolution = f'{resolution_minutes}min'
    start, end = pd.Timestamp(start).ceil(resolution), pd.Timestamp(end).floor(resolution)
    clock = set(pd.date_range(start, end, freq=resolution, inclusive='left')) if start < end else set()
    rows = stored.loc[stored.metric == metric].copy()
    if sources is not None:
        rows = rows.loc[rows.source_product.isin(sources)] if 'source_product' in rows else rows.iloc[:0]
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


def fetch_missing(pending, policies, adapters, *, stored_gaps=None):
    """Each priority round batches metrics sharing a provider request."""
    records, attempts, cache = [], [], {}
    pending = {metric: set(hours) for metric, hours in pending.items()}
    for rank in range(max(len(policies[m].sources.historical) for m in pending)):
        products = {}
        for metric, hours in pending.items():
            sources = policies[metric].sources.historical
            if not hours or rank >= len(sources):
                continue
            product = sources[rank]
            if stored_gaps is not None:
                hours.intersection_update(stored_gaps[metric, product])
            if hours:
                products.setdefault(product, []).append(metric)
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
                        received_at = datetime.now(UTC)
                        cache.setdefault(product, []).append((start, end, frame, received_at))
                        attempt['downloaded_measurements'] = len(frame)
                    else:
                        frame, received_at = cached[2:]
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
                        slot = observed.floor(f'{policies[row.metric].resolution_minutes}min')
                        # Hourly products must return hourly samples. Daily and
                        # three-hour coverage keeps the original observation time
                        # (for example GFZ noon) instead of relabeling it.
                        if ((policies[row.metric].resolution_minutes == 60 and observed != slot)
                                or not start <= observed < end or slot not in pending[row.metric]
                                or pd.isna(row.value) or not definition.accepts(row.value)):
                            continue
                        records.append({'metric': row.metric, 'value': row.value,
                                        'observed_at': observed, 'source_product': product,
                                        'received_at': received_at, 'spacecraft': getattr(row, 'spacecraft', None)})
                        if definition.coverage == 'latest':
                            pending[row.metric].clear()
                        else:
                            pending[row.metric].remove(slot)
                        accepted += 1
                    attempt['accepted_measurements'] = accepted
                except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                    attempt['error'] = str(exc)
                attempts.append(attempt)
    return pd.DataFrame(records, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at', 'spacecraft']), attempts


async def backfill_numeric(session, config: ClioObservations, metrics, *, now=None, raise_on_failure=True):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(metric not in config.observations or OBSERVATIONS[metric].kind != 'numeric'
                          or not config.observations[metric].sources.historical for metric in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    end = now.astimezone(UTC)
    ends = {m: pd.Timestamp(end).floor(f'{config.observations[m].resolution_minutes}min').to_pydatetime() for m in metrics}
    starts = {m: ends[m] - timedelta(days=config.observations[m].backfill_days) for m in metrics}
    earliest = min(starts.values())
    latest = max(ends.values())
    sources = {m: config.observations[m].sources.historical for m in metrics}
    products = list(dict.fromkeys(product for names in sources.values() for product in names))

    def gaps(stored, metric, allowed):
        policy = config.observations[metric]
        return missing_slots(stored, metric, starts[metric], ends[metric], sources=allowed,
                             freshness_minutes=policy.freshness_minutes, resolution_minutes=policy.resolution_minutes)

    stored = await load_measurements(session, since=earliest, until=latest, source_products=products, metrics=metrics)
    stored_gaps = {(m, product): gaps(stored, m, [product]) for m in metrics for product in sources[m]}
    pending = {m: stored_gaps[m, names[0]] for m, names in sources.items()}
    logger.info('Numeric backfill missing slots: %s', {
        m: min(1, len(slots)) if OBSERVATIONS[m].coverage == 'latest' else len(slots)
        for m, slots in pending.items()})
    if any(pending.values()):
        frame, attempts = await asyncio.to_thread(fetch_missing, pending, config.observations,
                                                  historical_adapters(), stored_gaps=stored_gaps)
    else:
        frame, attempts = pd.DataFrame(columns=['metric', 'value', 'observed_at']), []
    if raise_on_failure and frame.empty and any('error' in attempt for attempt in attempts):
        raise RuntimeError('No gaps filled; source errors: ' + '; '.join(
            f"{a['product']}: {a['error']}" for a in attempts if 'error' in a))
    written = await write_measurements(session, historical_records(frame), policy=WritePolicy.REPAIR_INVALID)
    # Reload after an attempted write so the report reflects persisted rows.
    if not frame.empty:
        stored = await load_measurements(session, since=earliest, until=latest, source_products=products, metrics=metrics)
    await session.commit()
    remaining = {m: (min(1, len(slots)) if OBSERVATIONS[m].coverage == 'latest' else len(slots))
                 for m in metrics for slots in [gaps(stored, m, sources[m])]}
    primary = {m: len(gaps(stored, m, [names[0]])) for m, names in sources.items()}
    failed_metrics = {m for attempt in attempts if 'error' in attempt for m in attempt['metrics'] if remaining[m]}
    return {'status': 'complete' if not any(remaining.values()) else 'partial',
            'ranges': {m: {'from': starts[m].isoformat(), 'to_exclusive': ends[m].isoformat(),
                          'resolution_minutes': config.observations[m].resolution_minutes} for m in metrics},
            'downloaded_measurements': sum(attempt.get('downloaded_measurements', 0) for attempt in attempts),
            'selected_measurements': len(frame), 'stored_measurements': written,
            'missing_primary_slots': primary,
            'missing_observed_slots': remaining,
            'missing_observed_hours': {m: count for m, count in remaining.items() if config.observations[m].resolution_minutes == 60},
            'failed_metrics': sorted(failed_metrics), 'source_attempts': attempts}
