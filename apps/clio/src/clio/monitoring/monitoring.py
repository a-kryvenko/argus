"""Read-only model input freshness alongside native collector diagnostics."""
from datetime import UTC, datetime
from sqlalchemy import select
from clio.db.models import Measurement, ScheduledJob
from clio.ingestion.products import OBSERVATIONS
from clio.monitoring.status import source_status

async def monitoring_status(session):
    now = datetime.now(UTC)
    result = await source_status(session, now)
    measurements = []
    # Monitor native numeric observations, not the legacy forecast input schema
    # (which still contains derived s10/m10/y10 indices without source policies).
    for metric, definition in OBSERVATIONS.items():
        if definition.kind != 'numeric':
            continue
        record = await session.scalar(select(Measurement).where(Measurement.metric == metric)
                                      .order_by(Measurement.observed_at.desc()).limit(1))
        received_at = record.received_at if record else None
        latest = record.observed_at if record else None
        age = (now - latest).total_seconds() if latest else None
        threshold = definition.max_age.total_seconds()
        measurements.append({'metric': metric, 'latest_observation_at': latest,
            'received_at': received_at,
            'age_seconds': age, 'stale_after_seconds': threshold,
            'status': 'unavailable' if record is None or record.value is None or record.quality in ('flagged', 'missing') else 'future' if age < -300 else 'delayed' if age > threshold else 'fresh'})
    job = await session.get(ScheduledJob, 'refresh')
    result['measurements'] = measurements
    result['last_refresh_completed_at'] = job.completed_at if job else None
    if any(m['status'] != 'fresh' for m in measurements):
        result['status'] = 'degraded'
    return result
