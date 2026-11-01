"""Availability and freshness derived from stored observations."""
from datetime import UTC, datetime, timedelta
from sqlalchemy import select
from clio.db.models.measurement import Measurement
from clio.ingestion.products import OBSERVATIONS
from clio.config import load_observation_config
from clio.observations.intervals import INTERVALS

async def observation_status(session):
    now = datetime.now(UTC)
    measurements = []
    # Monitor native numeric observations, not the legacy forecast input schema
    # (which still contains derived s10/m10/y10 indices without source policies).
    config = load_observation_config()
    numeric = {metric: policy for metric, policy in config.observations.items() if OBSERVATIONS[metric].kind == 'numeric'}
    records = {record.metric: record for record in await session.scalars(
        select(Measurement).where(Measurement.metric.in_(numeric))
        .distinct(Measurement.metric).order_by(Measurement.metric, Measurement.observed_at.desc()))}
    for metric, definition in numeric.items():
        record = records.get(metric)
        received_at = record.received_at if record else None
        latest = record.observed_at if record else None
        age = max(0, (now - latest - INTERVALS.get(metric, timedelta())).total_seconds()) if latest else None
        threshold = definition.freshness_minutes * 60
        measurements.append({'metric': metric, 'latest_observation_at': latest,
            'received_at': received_at,
            'age_seconds': age, 'stale_after_seconds': threshold,
            'status': 'unavailable' if record is None or record.value is None or not OBSERVATIONS[metric].accepts(record.value) or record.quality in ('flagged', 'missing') else 'future' if (latest - now).total_seconds() > 300 else 'delayed' if age > threshold else 'fresh'})
    return {'generated_at': now,
            'status': 'ok' if all(m['status'] == 'fresh' for m in measurements) else 'degraded',
            'measurements': measurements}
