"""Persistence and read models for unpropagated solar wind observations."""
import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from clio.providers.solar_wind_loader import FIELDS, SOURCES, fetch_records
from clio.db.models import Measurement
from clio.observations.native import wind_measurements, store_native
from clio.db.session import get_session_factory
from clio.db.locks import SOURCE_LOCKS
from clio.monitoring.specs import WIND_STALE_AFTER_SECONDS
from clio.monitoring.status import track_attempt
from clio.coverage import coverage

logger = logging.getLogger(__name__)
METADATA = {
    "bx": ("mag", "Bx", "nT", "GSM"),
    "by": ("mag", "By", "nT", "GSM"),
    "bz": ("mag", "Bz", "nT", "GSM"),
    "bt": ("mag", "Total magnetic field", "nT", None),
    "v": ("plasma", "Solar wind speed", "km/s", None),
    "n": ("plasma", "Proton density", "cm⁻³", None),
    "t": ("plasma", "Proton temperature", "K", None),
}


async def ingest_source(kind: str) -> None:
    async with get_session_factory()() as session:
        # Transaction-scoped lock also prevents duplicate collectors on multiple hosts.
        lock = await session.execute(text("SELECT pg_try_advisory_xact_lock(:key)"),
                                     {"key": SOURCE_LOCKS[f"solar_wind_{kind}"]})
        if not lock.scalar_one():
            logger.info("Solar wind %s ingestion is already running", kind)
            return
        async with track_attempt(f'solar_wind_{kind}', session, get_session_factory()) as attempt:
            records = await asyncio.to_thread(fetch_records, kind)
            # Replay the entire source window: a latest-time cursor would miss
            # internal gaps, late publications and revisions after downtime.
            attempt.received(records)
            await store_native(session, wind_measurements(kind, records))
        logger.info("Solar wind %s: ingested %d source samples", kind, len(records))


async def refresh_solar_wind() -> None:
    results = await asyncio.gather(*(ingest_source(kind) for kind in FIELDS), return_exceptions=True)
    failed = []
    for kind, result in zip(FIELDS, results):
        if isinstance(result, BaseException):
            logger.error("Solar wind %s failed: %s", kind, result)
            failed.append(kind)
    if failed:
        raise RuntimeError(f"Solar wind ingestion failed for: {', '.join(failed)}")


def metadata(metric: str) -> dict:
    kind, label, unit, coordinates = METADATA[metric]
    return {"label": label, "unit": unit, "coordinate_system": coordinates,
            "source": "NOAA SWPC RTSW", "source_url": SOURCES[kind],
            "location": "L1", "time_basis": "measurement", "propagated": False,
            "resolution_seconds": 60, "aggregation": "source_1_minute",
            "selection": "NOAA active spacecraft", "stale_after_seconds": WIND_STALE_AFTER_SECONDS}


def sample(record: Measurement, metric: str) -> dict:
    return {"observed_at": record.observed_at, "received_at": record.received_at,
            "value": record.value, "spacecraft": record.spacecraft, "quality": record.quality,
            "provider_quality": record.provider_quality}


async def latest(session: AsyncSession, metrics: list[str], now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    series = {}
    for metric in metrics:
        statement = (select(Measurement).where(Measurement.metric == metric,
                     Measurement.source_product == f'swpc.rtsw_{METADATA[metric][0]}',
                     Measurement.observed_at <= now).order_by(Measurement.observed_at.desc()).limit(1))
        record = (await session.execute(statement)).scalars().first()
        point = sample(record, metric) if record is not None else None
        age = max(0, int((now - record.observed_at).total_seconds())) if record is not None else None
        series[metric] = {**metadata(metric), "latest": point, "age_seconds": age,
                          "status": "missing" if point is None or point["value"] is None else
                          "stale" if age > WIND_STALE_AFTER_SECONDS else "fresh"}
    return {"generated_at": now, "series": series}


async def history(session: AsyncSession, metrics: list[str], start: datetime, end: datetime) -> dict:
    statement = (select(Measurement).where(Measurement.metric.in_(metrics),
                 Measurement.source_product.in_(['swpc.rtsw_mag', 'swpc.rtsw_plasma']),
                 Measurement.observed_at >= start, Measurement.observed_at < end)
                 .order_by(Measurement.observed_at))
    series = {metric: {**metadata(metric), "points": []} for metric in metrics}
    for record in (await session.execute(statement)).scalars():
        series[record.metric]['points'].append(sample(record, record.metric))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, 60)
    return {"from": start, "to": end, "interval": "[from,to)", "gap_filling": "none", "series": series}
