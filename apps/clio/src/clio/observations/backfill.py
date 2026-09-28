"""Gap-only numeric backfill, using each observation's historical cadence."""
import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
from requests import RequestException

from clio.config import ClioObservations
from clio.ingestion.adapters import historical_adapters
from clio.ingestion.products import OBSERVATIONS
from clio.observations.store import (
    load_measurements, upsert_measurements, upsert_normalized_observations,
)

def missing_slots(stored, metric, start, end):
    """Raw records close their native slot; normalized rows are never read."""
    resolution = OBSERVATIONS[metric].resolution
    start, end = pd.Timestamp(start).ceil(resolution), pd.Timestamp(end).floor(resolution)
    clock = set(pd.date_range(start, end, freq=resolution, inclusive='left')) if start < end else set()
    rows = stored.loc[stored.metric == metric]
    present = set(pd.to_datetime(rows.observed_at, utc=True).dt.floor(resolution))
    return clock - present


def missing_hours(stored, metric, start, end):
    """Compatibility helper for the original hourly plasma slice."""
    return missing_slots(stored, metric, start, end)


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
    records, attempts = [], []
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
                attempt = {'product': product, 'metrics': metrics,
                           'from': start.isoformat(), 'to_exclusive': end.isoformat()}
                try:
                    frame = adapters[product].fetch(start, end)
                    received_at = datetime.now(UTC)
                    accepted = 0
                    for row in frame.sort_values('observed_at').itertuples(index=False):
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
                        pending[row.metric].remove(slot)
                        accepted += 1
                    attempt['accepted_measurements'] = accepted
                except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                    attempt['error'] = str(exc)
                attempts.append(attempt)
    return pd.DataFrame(records, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at']), attempts


async def backfill_selected(session, config: ClioObservations, metrics, *, now=None, start=None, end=None,
                            raise_on_failure=True):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(metric not in config.observations or OBSERVATIONS[metric].kind != 'numeric' for metric in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    if (start is None) != (end is None):
        raise ValueError('Specify both start and end')
    if start is not None:
        if (start.tzinfo is None or end.tzinfo is None or start >= end or end > now
                or any(t.minute or t.second or t.microsecond for t in (start.astimezone(UTC), end.astimezone(UTC)))):
            raise ValueError('Choose past timezone-aware whole hours with start before end')
    end = (end or now).astimezone(UTC)
    ends = {m: pd.Timestamp(end).floor(OBSERVATIONS[m].resolution).to_pydatetime() for m in metrics}
    starts = {m: min(pd.Timestamp(start).tz_convert('UTC').ceil(OBSERVATIONS[m].resolution).to_pydatetime(), ends[m])
              if start is not None else ends[m] - timedelta(days=config.observations[m].backfill.days)
              for m in metrics}
    earliest = min(starts.values())
    latest = max(ends.values())
    stored = await load_measurements(session, since=earliest, until=latest)
    pending = {m: missing_slots(stored, m, starts[m], ends[m]) for m in metrics}
    frame, attempts = await asyncio.to_thread(fetch_missing, pending, config.observations, historical_adapters())
    if raise_on_failure and frame.empty and any('error' in attempt for attempt in attempts):
        raise RuntimeError('No gaps filled; source errors: ' + '; '.join(
            f"{a['product']}: {a['error']}" for a in attempts if 'error' in a))
    await upsert_measurements(session, frame, track_receipt=False, replace_existing=False)
    # Reload after insert: concurrent or pre-existing observations always win.
    stored = await load_measurements(session, since=earliest, until=latest)
    normalized_count = 0
    if not frame.empty:
        from clio.observations.derived import normalize_measurements
        normalized = normalize_measurements(stored)
        await upsert_normalized_observations(session, normalized)
        normalized_count = len(normalized)
    await session.commit()
    remaining = {m: len(missing_slots(stored, m, starts[m], ends[m])) for m in metrics}
    failed_metrics = {m for attempt in attempts if 'error' in attempt for m in attempt['metrics'] if remaining[m]}
    return {'status': 'complete' if not any(remaining.values()) else 'partial',
            'ranges': {m: {'from': starts[m].isoformat(), 'to_exclusive': ends[m].isoformat(),
                          'resolution': OBSERVATIONS[m].resolution} for m in metrics},
            'downloaded_measurements': len(frame), 'normalized_hours': normalized_count,
            'missing_observed_slots': remaining,
            'missing_observed_hours': {m: count for m, count in remaining.items() if OBSERVATIONS[m].resolution == '1h'},
            'failed_metrics': sorted(failed_metrics), 'source_attempts': attempts}
