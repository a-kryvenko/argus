"""Collect selected live observations with product batching and gap fallback."""
import asyncio
from datetime import UTC, datetime, timedelta

import pandas as pd
from requests import RequestException

from clio.observations.schema import LIVE_SOURCE_DAYS
from clio.ingestion.products import OBSERVATIONS
from clio.ingestion.live_adapters import live_adapters
from clio.observations.store import upsert_measurements


async def fetch_live(metrics, policies, adapters, now):
    start = now - timedelta(days=LIVE_SOURCE_DAYS)
    pending = {m: set(pd.date_range(pd.Timestamp(start).ceil(OBSERVATIONS[m].live_resolution),
                                   pd.Timestamp(now).floor(OBSERVATIONS[m].live_resolution),
                                   freq=OBSERVATIONS[m].live_resolution)) for m in metrics}
    rows, attempts, cache = [], [], {}
    async def receive(key, adapter):
        try:
            cache[key] = await adapter.fetch(start, now)
        except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
            cache[key] = exc
    for rank in range(max(len(policies[m].sources.live) for m in metrics)):
        products = {}
        for metric in metrics:
            sources = policies[metric].sources.live
            if pending[metric] and rank < len(sources):
                products.setdefault(sources[rank], []).append(metric)
        unique = {id(adapters[product]): adapters[product] for product in products if id(adapters[product]) not in cache}
        await asyncio.gather(*(receive(key, adapter) for key, adapter in unique.items()))
        for product, selected in products.items():
            attempt = {'product': product, 'metrics': selected}
            adapter = adapters[product]
            key = id(adapter)
            try:
                if isinstance(cache[key], Exception):
                    raise cache[key]
                frame = cache[key]
                requested = {m: pending[m].copy() for m in selected}
                accepted = 0
                for row in frame.itertuples(index=False):
                    if row.metric not in requested:
                        continue
                    definition = OBSERVATIONS[row.metric]
                    observed = pd.Timestamp(row.observed_at)
                    if observed.tzinfo is None:
                        raise ValueError('Adapter returned a timezone-naive observation')
                    observed = observed.tz_convert('UTC')
                    slot = observed.floor(definition.live_resolution)
                    if not start <= observed <= now or slot not in requested[row.metric] or not definition.accepts(row.value):
                        continue
                    rows.append(dict(metric=row.metric, value=row.value, observed_at=observed,
                                     source_product=product, received_at=row.received_at))
                    pending[row.metric].discard(slot)
                    accepted += 1
                attempt['accepted_measurements'] = accepted
            except (RequestException, OSError, ValueError, KeyError, RuntimeError) as exc:
                attempt['error'] = str(exc)
            attempts.append(attempt)
    frame = pd.DataFrame(rows, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at'])
    latest = {m: frame.loc[frame.metric == m, 'observed_at'].max() for m in metrics}
    failed = [m for m in metrics if pd.isna(latest[m]) or latest[m] < now - OBSERVATIONS[m].max_age]
    return frame, {'status': 'partial' if failed else 'complete', 'failed_metrics': failed,
                   'latest_observed_at': {m: None if pd.isna(t) else t.isoformat() for m, t in latest.items()},
                   'missing_live_slots': {m: len(slots) for m, slots in pending.items()}, 'source_attempts': attempts}


async def collect_live(session, config, metrics, *, now=None, heartbeat=None):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(m not in config.observations or OBSERVATIONS[m].kind != 'numeric' for m in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.astimezone(UTC)
    frame, report = await fetch_live(metrics, config.observations, live_adapters(heartbeat), now)
    await upsert_measurements(session, frame, source_priorities={m: config.observations[m].sources.live for m in metrics})
    await session.commit()
    report['downloaded_measurements'] = len(frame)
    return report
