"""Collect selected live observations with product batching and gap fallback."""
import asyncio
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from functools import partial

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

import pandas as pd
from requests import RequestException

from clio.ingestion.products import OBSERVATIONS
from clio.ingestion.live import LIVE_LOADERS, COLUMNS
from clio.db.session import get_db_session
from clio.db.locks import SOURCE_LOCKS
from clio.monitoring.status import track_attempt
from clio.observations.measurements import store_native

SOURCE_IDS = {
    'swpc.rtsw_mag': 'solar_wind_mag',
    'swpc.rtsw_plasma': 'solar_wind_plasma',
    'swpc.kp': 'kp',
    'swpc.dst': 'dst',
}


async def collect_product(product, priorities, start, now):
    async with get_db_session() as session:
        source_id = SOURCE_IDS.get(product)
        if source_id is not None:
            lock = await session.execute(text('SELECT pg_try_advisory_xact_lock(:key)'),
                                         {'key': SOURCE_LOCKS[source_id]})
            if not lock.scalar_one():
                return pd.DataFrame(columns=COLUMNS)
        context = track_attempt(source_id, session, get_db_session) if source_id else nullcontext()
        async with context as attempt:
            frame = await asyncio.to_thread(LIVE_LOADERS[product], start, now)
            records = frame.astype(object).where(pd.notna(frame), None).to_dict('records')
            if attempt is not None:
                attempt.received(records)
            await store_native(session, records, source_priorities=priorities)
            if attempt is None:
                await session.commit()
        return frame


async def fetch_live(metrics, policies, fetchers, now, *, lookback_days=6):
    start = now - timedelta(days=lookback_days)
    pending = {m: set(pd.date_range(pd.Timestamp(start).ceil(OBSERVATIONS[m].live_resolution),
                                   pd.Timestamp(now).floor(OBSERVATIONS[m].live_resolution),
                                   freq=OBSERVATIONS[m].live_resolution)) for m in metrics}
    rows, attempts, cache = [], [], {}
    async def receive(key, fetch):
        try:
            cache[key] = await fetch(start, now)
        except (RequestException, OSError, ValueError, KeyError, RuntimeError, SQLAlchemyError) as exc:
            cache[key] = exc
    for rank in range(max(len(policies[m].sources.live) for m in metrics)):
        products = {}
        for metric in metrics:
            sources = policies[metric].sources.live
            if pending[metric] and rank < len(sources):
                products.setdefault(sources[rank], []).append(metric)
        unique = {id(fetchers[product]): fetchers[product] for product in products if id(fetchers[product]) not in cache}
        await asyncio.gather(*(receive(key, fetch) for key, fetch in unique.items()))
        for product, selected in products.items():
            attempt = {'product': product, 'metrics': selected}
            key = id(fetchers[product])
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
                        raise ValueError('Loader returned a timezone-naive observation')
                    observed = observed.tz_convert('UTC')
                    slot = observed.floor(definition.live_resolution)
                    if (not start <= observed <= now or slot not in requested[row.metric]
                            or getattr(row, 'quality', None) in ('missing', 'flagged')
                            or pd.isna(row.value) or not definition.accepts(row.value)):
                        continue
                    rows.append(dict(metric=row.metric, value=row.value, observed_at=observed,
                                     source_product=product, received_at=row.received_at))
                    pending[row.metric].discard(slot)
                    accepted += 1
                attempt['accepted_measurements'] = accepted
            except (RequestException, OSError, ValueError, KeyError, RuntimeError, SQLAlchemyError) as exc:
                attempt['error'] = str(exc)
            attempts.append(attempt)
    frame = pd.DataFrame(rows, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at'])
    latest = {m: frame.loc[frame.metric == m, 'observed_at'].max() for m in metrics}
    failed = [m for m in metrics if pd.isna(latest[m]) or latest[m] < now - OBSERVATIONS[m].max_age]
    return frame, {'status': 'partial' if failed else 'complete', 'failed_metrics': failed,
                   'latest_observed_at': {m: None if pd.isna(t) else t.isoformat() for m, t in latest.items()},
                   'missing_live_slots': {m: len(slots) for m, slots in pending.items()}, 'source_attempts': attempts}


async def collect_live(config, metrics, *, now=None):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(m not in config.observations or OBSERVATIONS[m].kind != 'numeric' for m in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.astimezone(UTC)
    priorities = {m: policy.sources.live for m, policy in config.observations.items()
                  if OBSERVATIONS[m].kind == 'numeric'}
    fetchers = {product: partial(collect_product, product, priorities) for product in LIVE_LOADERS}
    frame, report = await fetch_live(metrics, config.observations, fetchers, now,
                                     lookback_days=config.retention.live_days)
    report['downloaded_measurements'] = len(frame)
    return report
