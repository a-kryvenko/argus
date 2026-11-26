"""Collect selected live observations with product batching and gap fallback."""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from functools import partial

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

import pandas as pd
from requests import RequestException

from clio.providers import f107, geomagnetic, goes_flux, solar_wind
from clio.ingestion.utils.adapters import wide_to_measurements, clean_records
from clio.ingestion.utils.products import OBSERVATIONS, PRODUCTS
from clio.db.session import get_db_session
from clio.ingestion.utils.measurements import WritePolicy, write_measurements
from clio.db.models.measurement import INTERVALS

logger = logging.getLogger(__name__)

def load_wind(kind, start, now):
    return measurements(solar_wind.fetch_records(kind), start, now, f'swpc.rtsw_{kind}')


def load_geomagnetic(metric, start, now):
    return measurements(geomagnetic.fetch_records(metric), start, now, f'swpc.{metric}')


def load_goes_flux(product, start, now):
    return measurements(goes_flux.fetch_records(product), start, now, product)


def measurements(dataset, start, now, product):
    received = datetime.now(UTC)
    frame = clean_records(wide_to_measurements(dataset), start,
                          pd.Timestamp(now) + pd.Timedelta(microseconds=1), PRODUCTS[product].metrics)
    return frame.drop_duplicates(['metric', 'observed_at'], keep='last').assign(
        received_at=received, source_product=product)


def load_f107(start, now):
    return measurements(f107.fetch_f107(), start, now, 'swpc.f107')


def load_celestrak_f107(start, now):
    from clio.providers.celestrak import fetch_celestrak_f107
    return measurements(fetch_celestrak_f107(), start, now, 'celestrak.f107')


def load_solfsmy_means(start, now):
    from clio.providers.solfsmy import fetch_solfsmy_means
    return measurements(fetch_solfsmy_means(), start, now, 'solfsmy.means')


LIVE_LOADERS = {
    'swpc.rtsw_mag': partial(load_wind, 'mag'),
    'swpc.rtsw_plasma': partial(load_wind, 'plasma'),
    'swpc.kp': partial(load_geomagnetic, 'kp'),
    'swpc.dst': partial(load_geomagnetic, 'dst'),
    'goes.xrs': partial(load_goes_flux, 'goes.xrs'),
    'goes.protons': partial(load_goes_flux, 'goes.protons'),
    'swpc.f107': load_f107,
    'celestrak.f107': load_celestrak_f107,
    'solfsmy.means': load_solfsmy_means,
}


@asynccontextmanager
async def product_session(product):
    """Hold the source lock until this product's write commits."""
    async with get_db_session() as session:
        lock_key = PRODUCTS[product].lock
        if lock_key is not None:
            lock = await session.execute(text('SELECT pg_try_advisory_xact_lock(:key)'),
                                         {'key': lock_key})
            if not lock.scalar_one():
                yield None
                return
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


def select_native_measurements(frame, product, priorities):
    allowed = {metric for metric, sources in priorities.items() if product in sources}
    return frame.loc[frame.metric.isin(allowed)]


def select_live_samples(frame, seen, first_slots, policies, now, product):
    rows = []
    covered = {metric: set() for metric in seen}
    for row in frame.itertuples(index=False):
        if row.metric not in seen:
            continue
        observed = pd.Timestamp(row.observed_at)
        if observed.tzinfo is None:
            raise ValueError('Loader returned a timezone-naive observation')
        observed = observed.tz_convert('UTC')
        slot = observed.floor(f'{policies[row.metric].live_resolution_minutes}min')
        if (observed > now or slot < first_slots[row.metric] or slot in seen[row.metric]
                or pd.isna(row.value) or not OBSERVATIONS[row.metric].accepts(row.value)):
            continue
        rows.append(dict(metric=row.metric, value=row.value, observed_at=observed,
                         source_product=product, received_at=row.received_at))
        covered[row.metric].add(slot)
    return rows, covered


async def fetch_live(metrics, policies, fetchers, now, *, lookback_days=6):
    start = now - timedelta(days=lookback_days)
    first_slots, expected = {}, {}
    covered_slots = {metric: set() for metric in metrics}
    for metric in metrics:
        step = pd.Timedelta(minutes=policies[metric].live_resolution_minutes)
        first_slots[metric] = pd.Timestamp(start).ceil(step)
        last = pd.Timestamp(now).floor(step)
        expected[metric] = max(0, (last - first_slots[metric]) // step + 1)
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
            if len(covered_slots[metric]) < expected[metric] and rank < len(sources):
                products.setdefault(sources[rank], []).append(metric)
        unique = {id(fetchers[product]): fetchers[product] for product in products if id(fetchers[product]) not in cache}
        await asyncio.gather(*(receive(key, fetch) for key, fetch in unique.items()))
        for product, selected in products.items():
            attempt = {'product': product, 'metrics': selected}
            key = id(fetchers[product])
            try:
                if isinstance(cache[key], Exception):
                    raise cache[key]
                samples, covered = select_live_samples(
                    cache[key], {metric: covered_slots[metric] for metric in selected}, first_slots, policies, now, product,
                )
                rows.extend(samples)
                for metric, slots in covered.items():
                    covered_slots[metric].update(slots)
                attempt['accepted_measurements'] = len(samples)
            except (RequestException, OSError, ValueError, KeyError, RuntimeError, SQLAlchemyError) as exc:
                logger.warning('Live source %s failed', product, exc_info=True)
                attempt['error'] = str(exc)
            attempts.append(attempt)
    frame = pd.DataFrame(rows, columns=['metric', 'value', 'observed_at', 'source_product', 'received_at'])
    latest = {m: frame.loc[frame.metric == m, 'observed_at'].max() for m in metrics}
    failed = [m for m in metrics if pd.isna(latest[m]) or latest[m] + INTERVALS.get(m, timedelta()) < now - timedelta(minutes=policies[m].freshness_minutes)]
    return frame, {'status': 'partial' if failed else 'complete', 'failed_metrics': failed,
                   'latest_observed_at': {m: None if pd.isna(t) else t.isoformat() for m, t in latest.items()},
                   'missing_live_slots': {m: expected[m] - len(covered_slots[m]) for m in metrics},
                   'source_attempts': attempts}


async def collect_numeric(config, metrics, *, now=None):
    metrics = list(dict.fromkeys(metrics))
    if not metrics or any(m not in config.observations or OBSERVATIONS[m].kind != 'numeric' for m in metrics):
        raise ValueError('Select observations configured in clio.observations')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.astimezone(UTC)
    priorities = {m: policy.sources.live for m, policy in config.observations.items()
                  if OBSERVATIONS[m].kind == 'numeric'}
    downloaded = 0
    stored = 0

    batches = {}

    async def fetch_product(product, start, end):
        nonlocal downloaded
        frame = await asyncio.to_thread(LIVE_LOADERS[product], start, end)
        downloaded += len(frame)
        native = select_native_measurements(frame, product, priorities)
        batches[product] = native
        return native

    fetchers = {product: partial(fetch_product, product) for product in LIVE_LOADERS}
    lookback = timedelta(days=config.retention.live_days)
    for metric in metrics:
        freshness = timedelta(minutes=config.observations[metric].freshness_minutes)
        lookback = max(lookback, freshness + INTERVALS.get(metric, timedelta()))
    lookback_days = lookback / timedelta(days=1)
    frame, report = await fetch_live(metrics, config.observations, fetchers, now, lookback_days=lookback_days)
    failed_products = {}
    for product, batch in batches.items():
        if batch.empty:
            continue
        try:
            async with product_session(product) as session:
                if session is None:
                    failed_products[product] = 'Another worker is committing this product'
                    continue
                records = batch.astype(object).where(pd.notna(batch), None).to_dict('records')
                written = await write_measurements(session, records, policy=WritePolicy.LIVE)
                await session.commit()
                stored += written
        except SQLAlchemyError as exc:
            failed_products[product] = str(exc)
            logger.warning('Live source %s write failed', product, exc_info=True)
    if failed_products:
        failed = set(report['failed_metrics'])
        for attempt in report['source_attempts']:
            if attempt['product'] in failed_products:
                attempt['error'] = failed_products[attempt['product']]
                failed.update(attempt['metrics'])
        frame = frame.loc[~frame.source_product.isin(failed_products)]
        report['status'] = 'partial'
        report['failed_metrics'] = sorted(failed)
        latest = frame.groupby('metric').observed_at.max().to_dict()
        report['latest_observed_at'] = {m: latest[m].isoformat() if m in latest else None for m in metrics}
    report['downloaded_measurements'] = downloaded
    report['selected_measurements'] = len(frame)
    report['stored_measurements'] = stored
    return report
