"""Persistence and read models for unpropagated solar wind observations."""
import asyncio
import logging
from datetime import UTC, datetime

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from clio.dataloaders.solar_wind_loader import FIELDS, SOURCES, fetch_records
from app.db.models import SolarWindObservation
from app.db.session import get_session_factory
from app.services.collection_status import track_attempt
from app.services.history_coverage import coverage

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
STALE_AFTER_SECONDS = 600


async def ingest_source(kind: str) -> None:
    async with get_session_factory()() as session:
        # Transaction-scoped lock also prevents duplicate collectors on multiple hosts.
        lock = await session.execute(text("SELECT pg_try_advisory_xact_lock(:key)"),
                                     {"key": 730100 if kind == "mag" else 730101})
        if not lock.scalar_one():
            logger.info("Solar wind %s ingestion is already running", kind)
            return
        async with track_attempt(f'solar_wind_{kind}', session, get_session_factory()) as attempt:
            records = await asyncio.to_thread(fetch_records, kind)
            # Replay the entire source window: a latest-time cursor would miss
            # internal gaps, late publications and revisions after downtime.
            attempt.received(records)
            for offset in range(0, len(records), 500):
                statement = insert(SolarWindObservation).values(records[offset:offset + 500])
                statement = statement.on_conflict_do_update(
                    index_elements=["kind", "observed_at", "spacecraft"],
                    set_={name: statement.excluded[name] for name in ("active", "received_at", "values", "raw")},
                    # Polling an unchanged point must not make its receipt time newer.
                    where=SolarWindObservation.raw.is_distinct_from(statement.excluded.raw),
                )
                await session.execute(statement)
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
            "selection": "NOAA active spacecraft", "stale_after_seconds": STALE_AFTER_SECONDS}


def sample(record: SolarWindObservation, metric: str) -> dict:
    value = record.values.get(metric)
    provider_quality = record.raw.get("overall_quality")
    quality = "missing" if value is None else "flagged" if provider_quality not in (None, 0) else "unverified"
    return {"observed_at": record.observed_at, "received_at": record.received_at,
            "value": value, "spacecraft": record.spacecraft, "quality": quality,
            "provider_quality": provider_quality}


async def latest(session: AsyncSession, metrics: list[str], now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    records = {}
    for kind in sorted({METADATA[metric][0] for metric in metrics}):
        statement = (select(SolarWindObservation)
                     .where(SolarWindObservation.kind == kind,
                            SolarWindObservation.active.is_(True), SolarWindObservation.observed_at <= now)
                     .order_by(SolarWindObservation.observed_at.desc(),
                               SolarWindObservation.received_at.desc(), SolarWindObservation.spacecraft)
                     .limit(1))
        record = (await session.execute(statement)).scalars().first()
        if record is not None:
            records[kind] = record
    series = {}
    for metric in metrics:
        record = records.get(METADATA[metric][0])
        point = sample(record, metric) if record is not None else None
        age = max(0, int((now - record.observed_at).total_seconds())) if record is not None else None
        series[metric] = {**metadata(metric), "latest": point, "age_seconds": age,
                          "status": "missing" if point is None or point["value"] is None else
                          "stale" if age > STALE_AFTER_SECONDS else "fresh"}
    return {"generated_at": now, "series": series}


async def history(session: AsyncSession, metrics: list[str], start: datetime, end: datetime) -> dict:
    kinds = {METADATA[metric][0] for metric in metrics}
    statement = (select(SolarWindObservation)
                 .where(SolarWindObservation.active.is_(True), SolarWindObservation.kind.in_(kinds),
                        SolarWindObservation.observed_at >= start, SolarWindObservation.observed_at < end)
                 .distinct(SolarWindObservation.kind, SolarWindObservation.observed_at)
                 .order_by(SolarWindObservation.kind, SolarWindObservation.observed_at,
                           SolarWindObservation.received_at.desc(), SolarWindObservation.spacecraft))
    records = list((await session.execute(statement)).scalars())
    series = {metric: {**metadata(metric), "points": []} for metric in metrics}
    for record in records:
        for metric in metrics:
            if METADATA[metric][0] == record.kind:
                series[metric]["points"].append(sample(record, metric))
    for metric, item in series.items():
        item['coverage'] = coverage(item['points'], start, end, 60)
    return {"from": start, "to": end, "interval": "[from,to)", "gap_filling": "none", "series": series}
